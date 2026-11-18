#!/usr/bin/env python3
"""Single-step GRPO and Boltzmann Q-distillation guided by Stockfish."""

import argparse
import json
import os
import random
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import chess
import chess.engine
import numpy as np
import torch
import torch.nn.functional as F

POST_TRAINING_DIR = Path(__file__).resolve().parent
REPO_ROOT = POST_TRAINING_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(POST_TRAINING_DIR))

from chess_hw import CONFIG, ChessTokenizer, Engine, ModelPlayer, load_model, save_model
from chess_hw.engine import MATE
from chess_hw.match import load_openings
from eval_local import evaluate_model_on_positions, load_or_create_val_positions
from inspect_repo import resolve_stockfish_path
from training_tracker import TrainingTracker, configure_nvidia_gpu, current_gpu_mem_mb

TOK = ChessTokenizer()
_RL_WORKER = {}


def _init_rl_worker(stockfish_path, sf_depth, sf_time, teacher_level):
    torch.set_num_threads(1)
    _RL_WORKER["tok"] = ChessTokenizer()
    _RL_WORKER["sf_depth"] = sf_depth
    _RL_WORKER["sf_time"] = sf_time
    _RL_WORKER["teacher"] = Engine(level=teacher_level, seed=123)
    _RL_WORKER["sf"] = None

    if stockfish_path and Path(stockfish_path).is_file():
        try:
            eng = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            eng.configure({"Threads": 1, "Hash": 16})
            _RL_WORKER["sf"] = eng
        except Exception:
            _RL_WORKER["sf"] = None


def _evaluate_all_legal_moves_cp(board):
    tok = _RL_WORKER.get("tok", TOK)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return {}, False

    sf = _RL_WORKER.get("sf")
    legal_id_to_cp = {}
    has_mate = False

    for mv in legal_moves:
        board.push(mv)
        is_mate = board.is_checkmate()
        board.pop()
        if is_mate:
            legal_id_to_cp[tok.token_to_id[board.uci(mv)]] = 10000.0
            has_mate = True

    if sf is not None:
        try:
            limit = chess.engine.Limit(
                depth=_RL_WORKER["sf_depth"],
                time=_RL_WORKER["sf_time"] if _RL_WORKER["sf_time"] > 0 else None,
            )
            infos = sf.analyse(board, limit, multipv=len(legal_moves))
            if isinstance(infos, dict):
                infos = [infos]
            for info in infos:
                pv = info.get("pv")
                if not pv:
                    continue
                tid = tok.token_to_id[board.uci(pv[0])]
                if tid in legal_id_to_cp and legal_id_to_cp[tid] >= 9999.0:
                    continue
                pov_score = info["score"].pov(board.turn)
                if pov_score.is_mate() and pov_score.mate() is not None and pov_score.mate() > 0:
                    legal_id_to_cp[tid] = 10000.0
                    has_mate = True
                else:
                    cp = pov_score.score(mate_score=9000)
                    legal_id_to_cp[tid] = float(cp if cp is not None else 0.0)
            if len(legal_id_to_cp) == len(legal_moves):
                return legal_id_to_cp, has_mate
        except Exception:
            pass

    teacher = _RL_WORKER["teacher"]
    b_copy = board.copy(stack=False)
    for mv in legal_moves:
        tid = tok.token_to_id[board.uci(mv)]
        if tid in legal_id_to_cp:
            continue
        b_copy.push(mv)
        if b_copy.is_checkmate():
            legal_id_to_cp[tid] = 10000.0
            has_mate = True
        else:
            sc = -teacher._search(b_copy, max(0, teacher.level - 1), -MATE - 1, MATE + 1, 1)
            if sc >= MATE - 50:
                legal_id_to_cp[tid] = 10000.0
                has_mate = True
            else:
                legal_id_to_cp[tid] = float(sc)
        b_copy.pop()

    return legal_id_to_cp, has_mate


def _rl_evaluate_position_task(task):
    prefix_ids, cand_ids, tau_reward, tau_q, raw_exp = task
    tok = _RL_WORKER.get("tok", TOK)

    board = chess.Board()
    for tid in prefix_ids[1:]:
        board.push_uci(tok.id_to_token[int(tid)])

    legal_id_to_cp, _ = _evaluate_all_legal_moves_cp(board)
    if not legal_id_to_cp:
        G = len(cand_ids)
        return {
            "rewards": np.zeros(G, dtype=np.float32),
            "advantages": np.zeros(G, dtype=np.float32),
            "legal_ids": np.zeros(0, dtype=np.int64),
            "q_boltzmann_probs": np.zeros(0, dtype=np.float32),
            "is_legal_flags": np.zeros(G, dtype=bool),
            "best_cp": 0.0,
        }

    best_cp = max(legal_id_to_cp.values())
    rewards = np.zeros(len(cand_ids), dtype=np.float32)
    is_legal_flags = np.zeros(len(cand_ids), dtype=bool)

    for i, aid in enumerate(cand_ids):
        aid_int = int(aid)
        if aid_int not in legal_id_to_cp:
            rewards[i] = -2.0
            is_legal_flags[i] = False
        else:
            is_legal_flags[i] = True
            cp_i = legal_id_to_cp[aid_int]
            if cp_i >= 9999.0:
                rewards[i] = 2.0
            else:
                delta_cp = min(0.0, float(cp_i - min(best_cp, 3000.0)))
                exp_val = float(np.exp(delta_cp / max(1.0, tau_reward)))
                rewards[i] = exp_val if raw_exp else (-0.5 + 1.5 * exp_val)

    mean_r = float(np.mean(rewards))
    std_r = float(np.std(rewards))
    advantages = np.zeros_like(rewards) if std_r < 1e-6 else (rewards - mean_r) / (std_r + 1e-5)

    legal_ids_arr = np.array(sorted(legal_id_to_cp.keys()), dtype=np.int64)
    q_vals = np.array([min(legal_id_to_cp[int(lid)], 2500.0) for lid in legal_ids_arr], dtype=np.float64)
    q_shifted = (q_vals - np.max(q_vals)) / max(1.0, float(tau_q))
    q_exp = np.exp(np.clip(q_shifted, -40.0, 0.0))
    q_probs = (q_exp / np.sum(q_exp)).astype(np.float32)

    return {
        "rewards": rewards,
        "advantages": advantages,
        "legal_ids": legal_ids_arr,
        "q_boltzmann_probs": q_probs,
        "is_legal_flags": is_legal_flags,
        "best_cp": float(best_cp),
    }


def collect_prefix_states(model, sft_samples, n_prefixes, seed):
    rng = random.Random(seed)
    prefixes = []
    openings = load_openings()

    if sft_samples:
        for _ in range(n_prefixes // 2):
            ids = rng.choice(sft_samples)["input_ids"].tolist()
            if len(ids) >= 8:
                cut = rng.randint(2, min(len(ids), CONFIG.n_ctx - 1))
                prefixes.append(ids[:cut])

    player = ModelPlayer(model)
    while len(prefixes) < n_prefixes:
        board = chess.Board()
        for u in rng.choice(openings):
            board.push_uci(u)
        opp = Engine(level=rng.choice([1, 2, 3]), seed=rng.randint(0, 100000))
        model_color = rng.random() < 0.5
        max_len = rng.randint(10, 180)
        while not board.is_game_over(claim_draw=True) and board.ply() < max_len:
            if board.turn == model_color:
                mv = player.choose(board).move or rng.choice(sorted(board.legal_moves, key=lambda m: m.uci()))
            else:
                mv = opp.best_move(board)
            if mv is None:
                break
            board.push(mv)
            if board.ply() >= 8 and rng.random() < 0.18:
                hist = [m.uci() for m in board.move_stack]
                if len(hist) < CONFIG.n_ctx:
                    prefixes.append(TOK.encode(hist))
                    if len(prefixes) >= n_prefixes:
                        break

    rng.shuffle(prefixes)
    return prefixes[:n_prefixes]


@torch.no_grad()
def sample_group_candidates(logits_last, group_size=8, strategy="top3_plus_sample", temperature=1.0):
    log_probs = F.log_softmax(logits_last.float(), dim=-1)
    probs = F.softmax(logits_last.float() / max(0.1, temperature), dim=-1)

    if strategy == "topk":
        actions = torch.topk(logits_last, k=group_size, dim=-1).indices
    elif strategy == "multinomial":
        actions = torch.multinomial(probs, num_samples=group_size, replacement=False)
    else:
        # Always include greedy top-3 moves in the group since eval uses top-3 decoding
        k_top = min(3, group_size)
        top_actions = torch.sort(logits_last, descending=True, stable=True).indices[:, :k_top]
        if group_size > k_top:
            probs_masked = probs.clone()
            probs_masked.scatter_(1, top_actions, 0.0)
            probs_masked = probs_masked.clamp_min(1e-8)
            probs_masked = probs_masked / probs_masked.sum(dim=-1, keepdim=True)
            sampled = torch.multinomial(probs_masked, num_samples=group_size - k_top, replacement=False)
            actions = torch.cat([top_actions, sampled], dim=-1)
        else:
            actions = top_actions

    old_log_probs = log_probs.gather(dim=-1, index=actions)
    return actions, old_log_probs


def run_rl_grpo_training(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device, use_amp, amp_dtype, gpu_info = configure_nvidia_gpu(
        enable_amp=not args.no_amp,
        auto_install_cuda=args.install_cuda_torch,
    )

    sft_path = Path(args.sft_model)
    if not (sft_path / "model.safetensors").exists() and not (sft_path.is_file() and sft_path.suffix == ".safetensors"):
        sft_path = REPO_ROOT / "base_model"
        print(f"[rl] sft checkpoint missing, falling back to {sft_path}")

    model = load_model(sft_path, device=device)
    ref_model = load_model(sft_path, device=device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    val_positions = load_or_create_val_positions(Path(args.val_data), count=args.val_count, workers=args.workers)
    sft_data_pt = Path(args.sft_data)
    sft_samples = torch.load(sft_data_pt, weights_only=False) if sft_data_pt.exists() else None

    out_dir = Path(args.out_dir)
    best_rl_dir = out_dir / "rl_best"
    latest_rl_dir = out_dir / "rl_latest"
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = TrainingTracker("rl_grpo", log_dir=POST_TRAINING_DIR / "logs", use_wandb=args.wandb, config=vars(args))

    init_eval = evaluate_model_on_positions(model, val_positions, batch_size=args.eval_batch_size, workers=args.workers)
    tracker.log_val({"step": 0, **init_eval})
    best_composite = init_eval["composite_score"]
    save_model(model, best_rl_dir)
    save_model(model, latest_rl_dir)

    print(
        f"[rl] mode={args.mode} G={args.group_size} beta_kl={args.beta_kl} | "
        f"init val_loss={init_eval['val_loss']:.4f} top1_leg={init_eval['top1_legal_rate']:.2%} "
        f"top3_leg={init_eval['top3_legal_rate']:.2%} sf_acc={init_eval['top1_sf_accuracy']:.2%}"
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.rl_steps * args.ppo_epochs), eta_min=args.lr * 0.1
    )

    sf_path = resolve_stockfish_path(auto_download=False)
    print(f"[rl] workers={args.workers}, engine={sf_path or f'Engine L{args.teacher_level}'}")

    history_log = []
    ctx = get_context("spawn")

    with ctx.Pool(
        processes=args.workers,
        initializer=_init_rl_worker,
        initargs=(sf_path, args.sf_depth, args.sf_time, args.teacher_level),
    ) as pool:
        for step in range(1, args.rl_steps + 1):
            t_step = time.time()

            model.eval()
            prefixes = collect_prefix_states(model, sft_samples, args.batch_size, args.seed + step)
            B = len(prefixes)
            max_len = max(len(p) for p in prefixes)
            x = torch.zeros(B, max_len, dtype=torch.long, device=device)
            last_idx = torch.zeros(B, dtype=torch.long, device=device)
            for b, p in enumerate(prefixes):
                x[b, : len(p)] = torch.tensor(p, dtype=torch.long, device=device)
                last_idx[b] = len(p) - 1

            rows = torch.arange(B, device=device)
            with torch.no_grad():
                logits_old = model(x)[rows, last_idx]
                ref_logits = ref_model(x)[rows, last_idx]
                ref_log_probs = F.log_softmax(ref_logits.float(), dim=-1)
                actions, old_log_probs = sample_group_candidates(
                    logits_old, group_size=args.group_size, strategy=args.sampling_strategy, temperature=args.temperature
                )

            worker_tasks = [
                (prefixes[b], actions[b].cpu().tolist(), args.tau_reward, args.tau_q, args.raw_exp_reward)
                for b in range(B)
            ]
            eval_results = pool.map(_rl_evaluate_position_task, worker_tasks, chunksize=max(1, B // (args.workers * 2)))

            advantages = torch.zeros(B, args.group_size, dtype=torch.float32, device=device)
            rewards_t = torch.zeros(B, args.group_size, dtype=torch.float32, device=device)
            legal_mask = torch.zeros(B, TOK.vocab_size, dtype=torch.bool, device=device)
            q_target_dist = torch.zeros(B, TOK.vocab_size, dtype=torch.float32, device=device)
            legal_cand_ratio = 0.0

            for b, res in enumerate(eval_results):
                rewards_t[b] = torch.from_numpy(res["rewards"]).to(device)
                advantages[b] = torch.from_numpy(res["advantages"]).to(device)
                leg_ids = torch.from_numpy(res["legal_ids"]).to(device)
                if len(leg_ids) > 0:
                    legal_mask[b, leg_ids] = True
                    q_target_dist[b, leg_ids] = torch.from_numpy(res["q_boltzmann_probs"]).to(device)
                legal_cand_ratio += float(np.mean(res["is_legal_flags"]))
            legal_cand_ratio /= max(1, B)

            model.train()
            step_metrics = {k: 0.0 for k in ("loss", "grpo_loss", "q_distill_loss", "kl_ref", "illegal_pen", "grad_norm")}

            for _ in range(args.ppo_epochs):
                with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                    logits_curr = model(x)[rows, last_idx].float()
                    log_probs_curr = F.log_softmax(logits_curr, dim=-1)
                    probs_curr = F.softmax(logits_curr, dim=-1)

                    cand_log_probs = log_probs_curr.gather(dim=-1, index=actions)
                    ratios = torch.exp((cand_log_probs - old_log_probs).clamp(-10.0, 10.0))
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * advantages
                    grpo_loss = -torch.min(surr1, surr2).mean()

                    has_legal = legal_mask.any(dim=-1)
                    if has_legal.any():
                        q_safe = q_target_dist.clamp_min(1e-12)
                        kl_q = (q_target_dist * (torch.log(q_safe) - log_probs_curr)).masked_fill(~legal_mask, 0.0).sum(dim=-1)
                        q_distill_loss = kl_q[has_legal].mean()

                        legal_log_mass = torch.logsumexp(log_probs_curr.masked_fill(~legal_mask, -1e9), dim=-1)
                        illegal_pen = (-legal_log_mass[has_legal]).mean()
                    else:
                        q_distill_loss = torch.zeros((), device=device)
                        illegal_pen = torch.zeros((), device=device)

                    kl_ref = (probs_curr * (log_probs_curr - ref_log_probs)).sum(dim=-1).mean()

                    if args.mode == "grpo":
                        loss = grpo_loss + args.beta_kl * kl_ref + args.lambda_legal * illegal_pen
                    elif args.mode == "q_distill":
                        loss = q_distill_loss + args.beta_kl * kl_ref + args.lambda_legal * illegal_pen
                    else:
                        loss = (
                            args.grpo_weight * grpo_loss
                            + args.q_distill_weight * q_distill_loss
                            + args.beta_kl * kl_ref
                            + args.lambda_legal * illegal_pen
                        )

                opt.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm).item())
                opt.step()
                sched.step()

                step_metrics["loss"] += float(loss.item()) / args.ppo_epochs
                step_metrics["grpo_loss"] += float(grpo_loss.item()) / args.ppo_epochs
                step_metrics["q_distill_loss"] += float(q_distill_loss.item()) / args.ppo_epochs
                step_metrics["kl_ref"] += float(kl_ref.item()) / args.ppo_epochs
                step_metrics["illegal_pen"] += float(illegal_pen.item()) / args.ppo_epochs
                step_metrics["grad_norm"] += grad_norm / args.ppo_epochs

            mean_reward = float(rewards_t.mean().item())
            top1_reward = float(rewards_t[:, 0].mean().item())
            curr_lr = float(sched.get_last_lr()[0])

            tracker.log_step({
                "step": step,
                "lr": round(curr_lr, 8),
                "grad_norm": round(step_metrics["grad_norm"], 4),
                "loss": round(step_metrics["loss"], 4),
                "grpo_loss": round(step_metrics["grpo_loss"], 4),
                "q_distill_loss": round(step_metrics["q_distill_loss"], 4),
                "kl_ref": round(step_metrics["kl_ref"], 4),
                "illegal_pen": round(step_metrics["illegal_pen"], 4),
                "mean_group_reward": round(mean_reward, 4),
                "top1_cand_reward": round(top1_reward, 4),
                "group_legal_ratio": round(legal_cand_ratio, 4),
                "gpu_mem_mb": current_gpu_mem_mb(),
            })

            if step % args.eval_every == 0 or step == args.rl_steps:
                save_model(model, latest_rl_dir)
                val_m = evaluate_model_on_positions(model, val_positions, batch_size=args.eval_batch_size, workers=args.workers)
                tracker.log_val({"step": step, **val_m})
                comp = val_m["composite_score"]
                is_best = False
                if comp >= best_composite and val_m["top3_legal_rate"] >= init_eval["top3_legal_rate"] - 0.005:
                    save_model(model, best_rl_dir)
                    best_composite = comp
                    is_best = True

                rec = {
                    "step": step,
                    "elapsed_sec": round(time.time() - t_step, 2),
                    "lr": round(curr_lr, 8),
                    "mean_group_reward": round(mean_reward, 4),
                    "top1_cand_reward": round(top1_reward, 4),
                    "group_legal_ratio": round(legal_cand_ratio, 4),
                    "train": {k: round(v, 4) for k, v in step_metrics.items()},
                    "val": val_m,
                    "saved_best": is_best,
                }
                history_log.append(rec)
                print(
                    f"[rl] step {step:>2}/{args.rl_steps} ({rec['elapsed_sec']:.1f}s) "
                    f"r_top1={top1_reward:+.3f} r_grp={mean_reward:+.3f} loss={step_metrics['loss']:.4f} | "
                    f"val_loss={val_m['val_loss']:.4f} top1_leg={val_m['top1_legal_rate']:.2%} "
                    f"top3_leg={val_m['top3_legal_rate']:.2%} sf_acc={val_m['top1_sf_accuracy']:.2%} acpl={val_m['acpl']:.1f}"
                    + (" [best]" if is_best else "")
                )
            else:
                print(
                    f"[rl] step {step:>2}/{args.rl_steps} ({time.time() - t_step:.1f}s) "
                    f"r_top1={top1_reward:+.3f} r_grp={mean_reward:+.3f} leg={legal_cand_ratio:.1%} "
                    f"loss={step_metrics['loss']:.4f} (grpo={step_metrics['grpo_loss']:.4f}, q={step_metrics['q_distill_loss']:.4f}, kl={step_metrics['kl_ref']:.4f})"
                )

    tracker.close()
    summary = {
        "gpu_info": gpu_info,
        "mode": args.mode,
        "best_rl_checkpoint": str(best_rl_dir / "model.safetensors"),
        "latest_rl_checkpoint": str(latest_rl_dir / "model.safetensors"),
        "logs": {"steps_csv": str(tracker.steps_csv), "val_csv": str(tracker.val_csv)},
        "initial_eval": init_eval,
        "best_composite_score": best_composite,
        "history": history_log,
    }
    (POST_TRAINING_DIR / "logs" / "rl_grpo_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[rl] done. best checkpoint -> {best_rl_dir / 'model.safetensors'}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_model", default=str(POST_TRAINING_DIR / "checkpoints" / "sft_best"))
    ap.add_argument("--sft_data", default=str(POST_TRAINING_DIR / "data" / "sft_datamix" / "train_datamix.pt"))
    ap.add_argument("--val_data", default=str(POST_TRAINING_DIR / "data" / "sft_datamix" / "val_positions.pt"))
    ap.add_argument("--out_dir", default=str(POST_TRAINING_DIR / "checkpoints"))
    ap.add_argument("--mode", choices=["grpo", "q_distill", "hybrid"], default="hybrid")
    ap.add_argument("--rl_steps", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--group_size", type=int, default=8)
    ap.add_argument("--ppo_epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--beta_kl", type=float, default=0.02)
    ap.add_argument("--lambda_legal", type=float, default=1.0)
    ap.add_argument("--grpo_weight", type=float, default=1.0)
    ap.add_argument("--q_distill_weight", type=float, default=1.0)
    ap.add_argument("--tau_reward", type=float, default=120.0)
    ap.add_argument("--tau_q", type=float, default=60.0)
    ap.add_argument("--raw_exp_reward", action="store_true")
    ap.add_argument("--sampling_strategy", choices=["top3_plus_sample", "multinomial", "topk"], default="top3_plus_sample")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--eval_batch_size", type=int, default=128)
    ap.add_argument("--val_count", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--sf_depth", type=int, default=10)
    ap.add_argument("--sf_time", type=float, default=0.03)
    ap.add_argument("--teacher_level", type=int, default=2)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--install_cuda_torch", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run_rl_grpo_training(args)


if __name__ == "__main__":
    main()
