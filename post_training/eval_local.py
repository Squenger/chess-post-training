#!/usr/bin/env python3
"""Local validation on 1000 annotated positions (losses, top-1/top-3 legality, ACPL)."""

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

from chess_hw import ChessTokenizer, Engine, load_model
from chess_hw.engine import MATE
from chess_hw.match import evaluate_vs_engine
from inspect_repo import resolve_stockfish_path
from training_tracker import configure_nvidia_gpu

TOK = ChessTokenizer()

SFT_GATE_TOP1_LEGAL = 0.92
SFT_GATE_TOP3_LEGAL = 0.98

_EVAL_WORKER = {}


def _init_eval_worker(stockfish_path, sf_depth, sf_time):
    _EVAL_WORKER["sf_depth"] = sf_depth
    _EVAL_WORKER["sf_time"] = sf_time
    _EVAL_WORKER["teacher"] = Engine(level=2, seed=0)
    _EVAL_WORKER["sf"] = None
    if stockfish_path and Path(stockfish_path).is_file():
        try:
            eng = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            eng.configure({"Threads": 1, "Hash": 16})
            _EVAL_WORKER["sf"] = eng
        except Exception:
            _EVAL_WORKER["sf"] = None


def _eval_move_cp_task(task):
    idx, fen, move_uci = task
    board = chess.Board(fen)
    board.push(chess.Move.from_uci(move_uci))

    if board.is_checkmate():
        return idx, 10000
    if board.is_game_over(claim_draw=True):
        return idx, 0

    sf = _EVAL_WORKER.get("sf")
    if sf is not None:
        try:
            limit = chess.engine.Limit(
                depth=max(1, _EVAL_WORKER["sf_depth"] - 1),
                time=_EVAL_WORKER["sf_time"] if _EVAL_WORKER["sf_time"] > 0 else None,
            )
            info = sf.analyse(board, limit)
            if isinstance(info, list):
                info = info[0]
            cp = -info["score"].pov(board.turn).score(mate_score=10000)
            return idx, int(cp if cp is not None else 0)
        except Exception:
            pass

    teacher = _EVAL_WORKER["teacher"]
    score = -teacher._search(board, max(0, teacher.level - 1), -MATE - 1, MATE + 1, 1)
    return idx, int(score)


@torch.inference_mode()
def evaluate_model_on_positions(
    model,
    val_positions,
    batch_size=128,
    workers=4,
    live_stockfish=False,
    sf_depth=10,
    sf_time=0.03,
    alpha=0.12,
    lambda_illegal=1.5,
):
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device

    n_pos = len(val_positions)
    top3_preds = []
    val_ce_sum = 0.0
    val_kl_uni_sum = 0.0
    val_ill_pen_sum = 0.0
    val_leg_prob_sum = 0.0

    for start in range(0, n_pos, batch_size):
        batch = val_positions[start : start + batch_size]
        B = len(batch)
        max_len = max(len(item["input_ids"]) for item in batch)
        x = torch.zeros(B, max_len, dtype=torch.long, device=device)
        last_idx = torch.zeros(B, dtype=torch.long, device=device)
        target_best = torch.zeros(B, dtype=torch.long, device=device)
        legal_mask = torch.zeros(B, TOK.vocab_size, dtype=torch.bool, device=device)

        for i, item in enumerate(batch):
            ids = item["input_ids"]
            x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            last_idx[i] = len(ids) - 1
            target_best[i] = int(item["best_move_id"])
            leg_ids = torch.tensor(item["legal_ids"], dtype=torch.long, device=device)
            legal_mask[i, leg_ids] = True

        logits_last = model(x)[torch.arange(B, device=device), last_idx].float()
        log_probs = F.log_softmax(logits_last, dim=-1)

        ce_b = F.cross_entropy(logits_last, target_best, reduction="sum").item()
        num_leg = legal_mask.sum(dim=-1).clamp_min(1).float()
        sum_logp_leg = log_probs.masked_fill(~legal_mask, 0.0).sum(dim=-1)
        kl_uni_b = (-torch.log(num_leg) - (sum_logp_leg / num_leg)).sum().item()

        leg_log_mass = torch.logsumexp(log_probs.masked_fill(~legal_mask, -1e9), dim=-1)
        leg_prob = torch.exp(leg_log_mass.clamp_max(0.0))
        ill_pen_b = (-leg_log_mass + (1.0 - leg_prob).clamp_min(0.0)).sum().item()

        val_ce_sum += ce_b
        val_kl_uni_sum += kl_uni_b
        val_ill_pen_sum += ill_pen_b
        val_leg_prob_sum += leg_prob.sum().item()

        order = torch.sort(logits_last.cpu(), descending=True, stable=True).indices[:, :3]
        top3_preds.extend(order.tolist())

    if was_training:
        model.train()

    val_ce_loss = val_ce_sum / max(1, n_pos)
    val_kl_uniform = val_kl_uni_sum / max(1, n_pos)
    val_illegal_pen = val_ill_pen_sum / max(1, n_pos)
    val_legal_prob = val_leg_prob_sum / max(1, n_pos)
    val_loss = (1.0 - alpha) * val_ce_loss + alpha * val_kl_uniform + lambda_illegal * val_illegal_pen

    attempts_used = []
    top1_sf_hits = []
    top3_sf_hits = []
    missing_cp_tasks = []
    played_cps = {}

    for idx, (item, cand_ids) in enumerate(zip(val_positions, top3_preds)):
        legal_set = set(item["legal_ids"])
        best_id = int(item["best_move_id"])

        attempt = 0
        chosen_id = None
        for att_i, cid in enumerate(cand_ids, start=1):
            if cid in legal_set:
                attempt = att_i
                chosen_id = cid
                break

        if chosen_id is None:
            rng = random.Random(f"val-fallback-{idx}")
            chosen_uci = rng.choice(sorted(item["legal_ucis"]))
        else:
            chosen_uci = TOK.id_to_token[chosen_id]

        attempts_used.append(attempt)
        top1_sf_hits.append(1 if cand_ids[0] == best_id else 0)
        top3_sf_hits.append(1 if best_id in cand_ids else 0)

        evals_map = item.get("move_evals_cp", {})
        if chosen_uci in evals_map and not live_stockfish:
            played_cps[idx] = int(evals_map[chosen_uci])
        elif chosen_uci == item["best_move_uci"] and not live_stockfish:
            played_cps[idx] = int(item.get("best_cp", 0))
        else:
            missing_cp_tasks.append((idx, item["fen"], chosen_uci))

    if missing_cp_tasks:
        sf_path = resolve_stockfish_path(auto_download=False) if live_stockfish else None
        if workers > 1 and len(missing_cp_tasks) >= 32:
            ctx = get_context("spawn")
            with ctx.Pool(
                processes=min(workers, 8),
                initializer=_init_eval_worker,
                initargs=(sf_path, sf_depth, sf_time),
            ) as pool:
                for r_idx, cp_val in pool.map(_eval_move_cp_task, missing_cp_tasks, chunksize=8):
                    played_cps[r_idx] = cp_val
        else:
            _init_eval_worker(sf_path, sf_depth, sf_time)
            for t in missing_cp_tasks:
                r_idx, cp_val = _eval_move_cp_task(t)
                played_cps[r_idx] = cp_val
            if _EVAL_WORKER.get("sf") is not None:
                try:
                    _EVAL_WORKER["sf"].quit()
                except Exception:
                    pass

    cpls = []
    slice_stats = {}

    for idx, item in enumerate(val_positions):
        best_cp = int(item.get("best_cp", 0))
        play_cp = int(played_cps.get(idx, best_cp))
        cpl = float(max(0, min(1000, best_cp - play_cp)))
        cpls.append(cpl)

        phase = item.get("phase", "middlegame")
        s_entry = slice_stats.setdefault(phase, {"top1_legal": [], "top3_legal": [], "top1_sf": [], "cpl": []})
        s_entry["top1_legal"].append(1.0 if attempts_used[idx] == 1 else 0.0)
        s_entry["top3_legal"].append(1.0 if attempts_used[idx] in (1, 2, 3) else 0.0)
        s_entry["top1_sf"].append(float(top1_sf_hits[idx]))
        s_entry["cpl"].append(cpl)

    top1_legal = sum(1 for a in attempts_used if a == 1) / max(n_pos, 1)
    top2_legal = sum(1 for a in attempts_used if a in (1, 2)) / max(n_pos, 1)
    top3_legal = sum(1 for a in attempts_used if a in (1, 2, 3)) / max(n_pos, 1)
    fallback_rate = sum(1 for a in attempts_used if a == 0) / max(n_pos, 1)
    top1_sf_acc = sum(top1_sf_hits) / max(n_pos, 1)
    top3_sf_acc = sum(top3_sf_hits) / max(n_pos, 1)
    mean_acpl = float(np.mean(cpls)) if cpls else 0.0

    rng_boot = np.random.default_rng(0)
    top3_arr = np.array([1.0 if a in (1, 2, 3) else 0.0 for a in attempts_used])
    cpl_arr = np.array(cpls)
    boot_idx = rng_boot.integers(0, max(n_pos, 1), size=(200, max(n_pos, 1)))
    top3_ci95 = np.percentile(top3_arr[boot_idx].mean(axis=1), [2.5, 97.5]).tolist() if n_pos > 0 else [0.0, 0.0]
    acpl_ci95 = np.percentile(cpl_arr[boot_idx].mean(axis=1), [2.5, 97.5]).tolist() if n_pos > 0 else [0.0, 0.0]

    slices_summary = {
        phase: {
            "count": len(d["cpl"]),
            "top1_legal_rate": round(float(np.mean(d["top1_legal"])), 4),
            "top3_legal_rate": round(float(np.mean(d["top3_legal"])), 4),
            "top1_sf_accuracy": round(float(np.mean(d["top1_sf"])), 4),
            "acpl": round(float(np.mean(d["cpl"])), 2),
        }
        for phase, d in sorted(slice_stats.items())
    }

    gate_passed = (top1_legal >= SFT_GATE_TOP1_LEGAL) and (top3_legal >= SFT_GATE_TOP3_LEGAL)
    composite_score = 100.0 * top3_legal + 50.0 * top1_legal + 30.0 * top1_sf_acc + 15.0 * top3_sf_acc - 0.05 * mean_acpl

    return {
        "n_positions": n_pos,
        "val_loss": round(float(val_loss), 4),
        "val_ce_loss": round(float(val_ce_loss), 4),
        "val_kl_uniform_legal": round(float(val_kl_uniform), 4),
        "val_illegal_penalty": round(float(val_illegal_pen), 4),
        "val_legal_prob": round(float(val_legal_prob), 4),
        "top1_legal_rate": float(top1_legal),
        "top2_legal_rate": float(top2_legal),
        "top3_legal_rate": float(top3_legal),
        "fallback_random_rate": float(fallback_rate),
        "top1_sf_accuracy": float(top1_sf_acc),
        "top3_sf_accuracy": float(top3_sf_acc),
        "acpl": mean_acpl,
        "top3_legal_ci95": [round(x, 4) for x in top3_ci95],
        "acpl_ci95": [round(x, 2) for x in acpl_ci95],
        "sft_gate_passed": bool(gate_passed),
        "sft_gate_thresholds": {"top1_legal_min": SFT_GATE_TOP1_LEGAL, "top3_legal_min": SFT_GATE_TOP3_LEGAL},
        "composite_score": round(float(composite_score), 4),
        "slices": slices_summary,
    }


def load_or_create_val_positions(val_path, count=1000, workers=4):
    val_path = Path(val_path)
    if val_path.exists():
        return torch.load(val_path, weights_only=False)
    print(f"[eval_local] generating {count} validation positions at {val_path}...")
    from prepare_data import build_sft_dataset
    build_sft_dataset(
        out_dir=val_path.parent,
        total_sequences=max(200, count // 2),
        val_positions_count=count,
        workers=workers,
        use_hf_dataset=False,
    )
    return torch.load(val_path, weights_only=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(REPO_ROOT / "base_model"))
    ap.add_argument("--val_data", default=str(POST_TRAINING_DIR / "data" / "sft_datamix" / "val_positions.pt"))
    ap.add_argument("--val_count", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--live_stockfish", action="store_true")
    ap.add_argument("--run_engine_match", action="store_true")
    ap.add_argument("--n_openings", type=int, default=25)
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device, _, _, _ = configure_nvidia_gpu(enable_amp=True)
    model = load_model(args.model, device=device)
    val_positions = load_or_create_val_positions(Path(args.val_data), count=args.val_count, workers=args.workers)

    t0 = time.time()
    m = evaluate_model_on_positions(
        model=model,
        val_positions=val_positions,
        batch_size=args.batch_size,
        workers=args.workers,
        live_stockfish=args.live_stockfish,
    )
    print(f"[eval_local] model={args.model} ({time.time() - t0:.1f}s, n={m['n_positions']})")
    print(f"  val_loss={m['val_loss']:.4f} (ce={m['val_ce_loss']:.4f}, kl_uni={m['val_kl_uniform_legal']:.4f}, ill_pen={m['val_illegal_penalty']:.4f})")
    print(f"  legal: top1={m['top1_legal_rate']:.2%}, top3={m['top3_legal_rate']:.2%}, fallback={m['fallback_random_rate']:.2%}")
    print(f"  sf_acc: top1={m['top1_sf_accuracy']:.2%}, top3={m['top3_sf_accuracy']:.2%}, acpl={m['acpl']:.1f}")
    for phase, s in m["slices"].items():
        print(f"    {phase:<15} n={s['count']:>4} top1={s['top1_legal_rate']:.1%} top3={s['top3_legal_rate']:.2%} sf={s['top1_sf_accuracy']:.1%} acpl={s['acpl']:.1f}")

    if args.run_engine_match:
        match = evaluate_vs_engine(args.model, args.levels, args.n_openings, seed=0, workers=args.workers)
        m["engine_match"] = match
        print(f"  engine match L{args.levels}: score={match['mean_score']:.3f} legal1={match['legal_first_try']:.2%} legal3={match['legal_within_3']:.2%}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(m, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
