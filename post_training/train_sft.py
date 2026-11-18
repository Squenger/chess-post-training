#!/usr/bin/env python3
"""Supervised fine-tuning for ChessGPT with legal-mask regularization and on-policy DAgger refresh."""

import argparse
import json
import math
import os
import random
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

POST_TRAINING_DIR = Path(__file__).resolve().parent
REPO_ROOT = POST_TRAINING_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(POST_TRAINING_DIR))

from chess_hw import ChessTokenizer, load_model, save_model
from chess_hw.match import load_openings
from eval_local import (
    SFT_GATE_TOP1_LEGAL,
    SFT_GATE_TOP3_LEGAL,
    evaluate_model_on_positions,
    load_or_create_val_positions,
)
from inspect_repo import resolve_stockfish_path
from prepare_data import _init_data_worker, _worker_process_task, build_sft_dataset
from training_tracker import TrainingTracker, configure_nvidia_gpu, current_gpu_mem_mb

TOK = ChessTokenizer()


class SFTDataset(Dataset):
    def __init__(self, samples):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def add_samples(self, new_samples):
        self.samples.extend(new_samples)


def collate_sft_batch(batch):
    B = len(batch)
    T = max(len(item["input_ids"]) for item in batch)
    V = TOK.vocab_size

    input_ids = torch.zeros(B, T, dtype=torch.long)
    target_ids = torch.full((B, T), -100, dtype=torch.long)
    weights = torch.zeros(B, T, dtype=torch.float32)
    legal_mask = torch.zeros(B, T, V, dtype=torch.bool)
    valid_mask = torch.zeros(B, T, dtype=torch.bool)
    pillars = torch.zeros(B, dtype=torch.long)

    for b, item in enumerate(batch):
        inp = item["input_ids"]
        L = len(inp)
        input_ids[b, :L] = torch.from_numpy(inp.astype(np.int64))
        target_ids[b, :L] = torch.from_numpy(item["target_ids"].astype(np.int64))
        weights[b, :L] = torch.from_numpy(item["weights"].astype(np.float32))
        valid_mask[b, :L] = True
        pillars[b] = int(item.get("pillar", 0))

        packed = item["legal_packed"]
        offsets = item["legal_offsets"]
        for t in range(L):
            start, end = int(offsets[t]), int(offsets[t + 1])
            if end > start:
                leg_ids = torch.from_numpy(packed[start:end].astype(np.int64))
                legal_mask[b, t, leg_ids] = True

    return input_ids, target_ids, weights, legal_mask, valid_mask, pillars


def compute_hybrid_sft_loss(
    logits,
    target_ids,
    weights,
    legal_mask,
    valid_mask,
    pillars,
    alpha=0.12,
    lambda_illegal=1.5,
    ref_logits=None,
    beta_ref=0.08,
):
    logits_f32 = logits.float()
    log_probs = F.log_softmax(logits_f32, dim=-1)

    B, T, V = logits_f32.shape
    ce_per_token = F.cross_entropy(
        logits_f32.view(B * T, V),
        target_ids.view(B * T),
        ignore_index=-100,
        reduction="none",
    ).view(B, T)

    active_ce_mask = (target_ids >= 0) & valid_mask
    eff_weights = weights * active_ce_mask.float()

    # Pillar 2 emphasizes legal move coverage (higher uniform smoothing weight)
    pillar_alpha = torch.where(
        pillars[:, None] == 2,
        torch.full((B, 1), min(0.45, alpha * 2.0), device=logits.device),
        torch.full((B, 1), alpha, device=logits.device),
    )

    weighted_ce = ((1.0 - pillar_alpha) * ce_per_token * eff_weights).sum() / eff_weights.sum().clamp_min(1.0)

    num_legal = legal_mask.sum(dim=-1).clamp_min(1).float()
    sum_logp_legal = log_probs.masked_fill(~legal_mask, 0.0).sum(dim=-1)
    kl_uniform_legal = -torch.log(num_legal) - (sum_logp_legal / num_legal)
    kl_uniform_loss = (pillar_alpha * kl_uniform_legal * valid_mask.float()).sum() / valid_mask.float().sum().clamp_min(1.0)

    legal_log_mass = torch.logsumexp(log_probs.masked_fill(~legal_mask, -1e9), dim=-1)
    illegal_prob_mass = (1.0 - torch.exp(legal_log_mass.clamp_max(0.0))).clamp_min(0.0)
    illegal_penalty_per_pos = -legal_log_mass + illegal_prob_mass
    illegal_penalty = (illegal_penalty_per_pos * valid_mask.float()).sum() / valid_mask.float().sum().clamp_min(1.0)

    if ref_logits is not None and beta_ref > 0.0:
        with torch.no_grad():
            ref_logp = F.log_softmax(ref_logits.float(), dim=-1)
            q_ref = F.softmax(ref_logp.masked_fill(~legal_mask, -1e9), dim=-1)
        log_q_model = (log_probs - legal_log_mass.unsqueeze(-1)).masked_fill(~legal_mask, 0.0)
        kl_ref_per_pos = (q_ref * (torch.log(q_ref.clamp_min(1e-12)) - log_q_model)).masked_fill(~legal_mask, 0.0).sum(dim=-1)
        kl_ref_loss = (kl_ref_per_pos * valid_mask.float()).sum() / valid_mask.float().sum().clamp_min(1.0)
    else:
        kl_ref_loss = torch.zeros((), device=logits.device)

    total_loss = weighted_ce + kl_uniform_loss + lambda_illegal * illegal_penalty + beta_ref * kl_ref_loss
    metrics = {
        "loss": float(total_loss.item()),
        "ce_best": float(weighted_ce.item()),
        "kl_uniform_legal": float(kl_uniform_loss.item()),
        "illegal_penalty": float(illegal_penalty.item()),
        "kl_ref": float(kl_ref_loss.item()),
        "mean_legal_prob": float(torch.exp(legal_log_mass.clamp_max(0.0))[valid_mask].mean().item()),
    }
    return total_loss, metrics


def collect_onpolicy_refresh_samples(model_path, n_games, seed, workers, sf_depth=10, sf_time=0.03, teacher_level=2):
    if n_games <= 0:
        return []
    openings = load_openings()
    sf_path = resolve_stockfish_path(auto_download=False)
    tasks = []
    for i in range(n_games):
        op = openings[(seed * 997 + i * 19) % len(openings)]
        pillar = 1 if (i % 3 != 0) else 2
        tasks.append(("onpolicy_tactical", (op, seed * 10000 + i, pillar)))

    out = []
    ctx = get_context("spawn")
    with ctx.Pool(
        processes=workers,
        initializer=_init_data_worker,
        initargs=(sf_path, sf_depth, sf_time, str(model_path), teacher_level),
    ) as pool:
        for res in pool.imap_unordered(_worker_process_task, tasks, chunksize=max(1, len(tasks) // (workers * 2))):
            if res is not None:
                out.append(res)
    return out


def build_optimizer_and_scheduler(model, lr, weight_decay, total_steps, warmup_ratio=0.08):
    decay_params = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    opt = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return opt, sched


def run_sft_training(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device, use_amp, amp_dtype, gpu_info = configure_nvidia_gpu(
        enable_amp=not args.no_amp,
        auto_install_cuda=args.install_cuda_torch,
    )

    data_dir = Path(args.data_dir)
    train_pt = data_dir / "train_datamix.pt"
    val_pt = data_dir / "val_positions.pt"

    if not train_pt.exists() or not val_pt.exists():
        print(f"[sft] dataset missing in {data_dir}, building...")
        build_sft_dataset(
            out_dir=data_dir,
            total_sequences=args.auto_prepare_sequences,
            val_positions_count=args.val_count,
            workers=args.workers,
            sf_depth=args.sf_depth,
            sf_time_limit=args.sf_time,
            teacher_level=args.teacher_level,
            use_hf_dataset=not args.no_hf,
            seed=args.seed,
        )

    samples = torch.load(train_pt, weights_only=False)
    val_positions = load_or_create_val_positions(val_pt, count=args.val_count, workers=args.workers)
    dataset = SFTDataset(samples)

    model = load_model(args.base_model, device=device)
    ref_model = load_model(args.base_model, device=device) if args.beta_ref > 0 else None
    if ref_model is not None:
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    out_dir = Path(args.out_dir)
    best_dir = out_dir / "sft_best"
    latest_dir = out_dir / "sft_latest"
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = TrainingTracker("sft", log_dir=POST_TRAINING_DIR / "logs", use_wandb=args.wandb, config=vars(args))

    base_eval = evaluate_model_on_positions(
        model=model,
        val_positions=val_positions,
        batch_size=args.eval_batch_size,
        workers=args.workers,
        alpha=args.alpha,
        lambda_illegal=args.lambda_illegal,
    )
    tracker.log_val({"step": 0, "epoch": 0, **base_eval})
    print(
        f"[sft] base: val_loss={base_eval['val_loss']:.4f} (ce={base_eval['val_ce_loss']:.4f}) | "
        f"top1_leg={base_eval['top1_legal_rate']:.2%} top3_leg={base_eval['top3_legal_rate']:.2%} | "
        f"sf_acc={base_eval['top1_sf_accuracy']:.2%} acpl={base_eval['acpl']:.1f}"
    )

    est_total_samples = len(dataset) + max(0, args.epochs - 1) * args.onpolicy_refresh_games
    steps_per_epoch = max(1, math.ceil(est_total_samples / args.batch_size))
    total_steps = steps_per_epoch * args.epochs

    opt, sched = build_optimizer_and_scheduler(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    best_composite = -1e9
    best_gate_passed = False
    history_log = []
    global_step = 0

    save_model(model, latest_dir)

    for epoch in range(args.epochs):
        t_ep = time.time()

        if epoch > 0 and args.onpolicy_refresh_games > 0:
            new_samples = collect_onpolicy_refresh_samples(
                model_path=latest_dir,
                n_games=args.onpolicy_refresh_games,
                seed=args.seed + epoch,
                workers=args.workers,
                sf_depth=args.sf_depth,
                sf_time=args.sf_time,
                teacher_level=args.teacher_level,
            )
            dataset.add_samples(new_samples)
            print(f"[sft] epoch {epoch + 1}: added {len(new_samples)} DAgger games (total={len(dataset)})")

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_sft_batch,
            num_workers=0,
            pin_memory=(device == "cuda"),
        )

        model.train()
        running = {k: 0.0 for k in ("loss", "ce_best", "kl_uniform_legal", "illegal_penalty", "kl_ref", "mean_legal_prob", "grad_norm")}
        n_batches = 0

        pbar = tqdm(loader, desc=f"sft ep {epoch + 1}/{args.epochs}", leave=False)
        for input_ids, target_ids, weights, legal_mask, valid_mask, pillars in pbar:
            global_step += 1
            input_ids = input_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            legal_mask = legal_mask.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)
            pillars = pillars.to(device, non_blocking=True)

            with torch.no_grad():
                ref_logits = ref_model(input_ids) if ref_model is not None else None

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                logits = model(input_ids)
                loss, batch_metrics = compute_hybrid_sft_loss(
                    logits=logits,
                    target_ids=target_ids,
                    weights=weights,
                    legal_mask=legal_mask,
                    valid_mask=valid_mask,
                    pillars=pillars,
                    alpha=args.alpha,
                    lambda_illegal=args.lambda_illegal,
                    ref_logits=ref_logits,
                    beta_ref=args.beta_ref,
                )

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm).item())
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm).item())
                opt.step()

            curr_lr = float(sched.get_last_lr()[0])
            sched.step()

            batch_metrics["grad_norm"] = round(grad_norm, 4)
            for k in running:
                running[k] += batch_metrics[k]
            n_batches += 1

            tracker.log_step({
                "step": global_step,
                "epoch": epoch + 1,
                "lr": round(curr_lr, 8),
                "grad_norm": round(grad_norm, 4),
                "loss": round(batch_metrics["loss"], 4),
                "ce_best": round(batch_metrics["ce_best"], 4),
                "kl_uniform_legal": round(batch_metrics["kl_uniform_legal"], 4),
                "illegal_penalty": round(batch_metrics["illegal_penalty"], 4),
                "kl_ref": round(batch_metrics["kl_ref"], 4),
                "mean_legal_prob": round(batch_metrics["mean_legal_prob"], 4),
                "gpu_mem_mb": current_gpu_mem_mb(),
            })
            pbar.set_postfix(loss=f"{batch_metrics['loss']:.3f}", ce=f"{batch_metrics['ce_best']:.3f}", lr=f"{curr_lr:.1e}")

        avg_train = {k: round(v / max(1, n_batches), 4) for k, v in running.items()}
        save_model(model, latest_dir)

        val_metrics = evaluate_model_on_positions(
            model=model,
            val_positions=val_positions,
            batch_size=args.eval_batch_size,
            workers=args.workers,
            alpha=args.alpha,
            lambda_illegal=args.lambda_illegal,
        )
        tracker.log_val({"step": global_step, "epoch": epoch + 1, **val_metrics})

        gate_ok = val_metrics["sft_gate_passed"]
        comp = val_metrics["composite_score"]

        saved_best = False
        if gate_ok and (not best_gate_passed or comp > best_composite):
            save_model(model, best_dir)
            best_composite = comp
            best_gate_passed = True
            saved_best = True
        elif not best_gate_passed and comp > best_composite:
            best_composite = comp
            if not (best_dir / "model.safetensors").exists():
                save_model(model, best_dir)

        ep_record = {
            "epoch": epoch + 1,
            "step": global_step,
            "elapsed_sec": round(time.time() - t_ep, 2),
            "lr_end": round(float(sched.get_last_lr()[0]), 8),
            "train": avg_train,
            "val": val_metrics,
            "saved_best_gate": saved_best,
        }
        history_log.append(ep_record)

        print(
            f"[sft] ep {epoch + 1}/{args.epochs} ({ep_record['elapsed_sec']:.1f}s) "
            f"lr={ep_record['lr_end']:.2e} train_loss={avg_train['loss']:.4f} -> "
            f"val_loss={val_metrics['val_loss']:.4f} top1_leg={val_metrics['top1_legal_rate']:.2%} "
            f"top3_leg={val_metrics['top3_legal_rate']:.2%} sf_acc={val_metrics['top1_sf_accuracy']:.2%} "
            f"acpl={val_metrics['acpl']:.1f}" + (" [best]" if saved_best else "")
        )

    tracker.close()
    summary = {
        "gpu_info": gpu_info,
        "best_checkpoint": str(best_dir / "model.safetensors"),
        "latest_checkpoint": str(latest_dir / "model.safetensors"),
        "sft_gate_passed": best_gate_passed,
        "best_composite_score": best_composite,
        "logs": {"steps_csv": str(tracker.steps_csv), "val_csv": str(tracker.val_csv)},
        "base_eval": base_eval,
        "history": history_log,
    }
    (POST_TRAINING_DIR / "logs" / "sft_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[sft] done. best checkpoint -> {best_dir / 'model.safetensors'}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default=str(REPO_ROOT / "base_model"))
    ap.add_argument("--data_dir", default=str(POST_TRAINING_DIR / "data" / "sft_datamix"))
    ap.add_argument("--out_dir", default=str(POST_TRAINING_DIR / "checkpoints"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--eval_batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.08)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.12)
    ap.add_argument("--lambda_illegal", type=float, default=1.5)
    ap.add_argument("--beta_ref", type=float, default=0.08)
    ap.add_argument("--onpolicy_refresh_games", type=int, default=400)
    ap.add_argument("--auto_prepare_sequences", type=int, default=2000)
    ap.add_argument("--val_count", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--sf_depth", type=int, default=12)
    ap.add_argument("--sf_time", type=float, default=0.04)
    ap.add_argument("--teacher_level", type=int, default=2)
    ap.add_argument("--no_hf", action="store_true")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--install_cuda_torch", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run_sft_training(args)


if __name__ == "__main__":
    main()
