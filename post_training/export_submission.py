#!/usr/bin/env python3
"""Export final safetensors weights, run head-to-head/engine benchmarks, and verify submission."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import chess
import numpy as np
import torch

POST_TRAINING_DIR = Path(__file__).resolve().parent
REPO_ROOT = POST_TRAINING_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(POST_TRAINING_DIR))

from chess_hw import ChessGPT, load_model, save_model, validate_state_dict
from chess_hw.checks import check_safetensors_file
from chess_hw.match import _pair_task, evaluate_vs_engine, run_tasks, score_for, select_openings
from chess_hw.model import reference_shapes


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_any_checkpoint(ckpt_path):
    if ckpt_path.is_dir():
        st_file = ckpt_path / "model.safetensors"
        if st_file.exists():
            return load_model(st_file, device="cpu")
        for ext in ("*.safetensors", "*.pt", "*.bin"):
            found = list(ckpt_path.glob(ext))
            if found:
                ckpt_path = found[0]
                break

    if ckpt_path.suffix == ".safetensors":
        return load_model(ckpt_path, device="cpu")

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    ref = reference_shapes()
    cleaned = {}
    for k in ref:
        if k in raw:
            cleaned[k] = raw[k].detach().to("cpu", torch.float32).contiguous()
        elif f"module.{k}" in raw:
            cleaned[k] = raw[f"module.{k}"].detach().to("cpu", torch.float32).contiguous()
    err = validate_state_dict(cleaned)
    if err:
        raise ValueError(f"Invalid checkpoint {ckpt_path}: {err}")
    model = ChessGPT()
    model.load_state_dict(cleaned, strict=True)
    return model.eval()


def compute_elo_difference(score):
    s = max(0.005, min(0.995, float(score)))
    return -400.0 * math.log10(1.0 / s - 1.0)


def evaluate_head_to_head(candidate_path, base_path, n_openings=50, seed=100, workers=4):
    oids = select_openings(seed, n_openings)
    tasks = []
    for oid in oids:
        tasks.append(("candidate", "base", oid, True))
        tasks.append(("candidate", "base", oid, False))

    records = run_tasks(_pair_task, tasks, {"candidate": str(candidate_path), "base": str(base_path)},
                        workers=workers, progress=True)

    wins, draws, losses = 0, 0, 0
    cand_dec, cand_a1, cand_a123, cand_fb = 0, 0, 0, 0
    base_dec, base_a1, base_a123, base_fb = 0, 0, 0, 0
    game_scores = []

    for r in records:
        cand_white = bool(r["meta"]["a_white"])
        s = score_for(r["result"], chess.WHITE if cand_white else chess.BLACK)
        game_scores.append(s)
        if s == 1.0:
            wins += 1
        elif s == 0.5:
            draws += 1
        else:
            losses += 1

        c_st = r["white_stats"] if cand_white else r["black_stats"]
        b_st = r["black_stats"] if cand_white else r["white_stats"]
        cand_dec += c_st["decisions"]
        cand_a1 += c_st["attempts"][0]
        cand_a123 += sum(c_st["attempts"])
        cand_fb += c_st["fallbacks"]

        base_dec += b_st["decisions"]
        base_a1 += b_st["attempts"][0]
        base_a123 += sum(b_st["attempts"])
        base_fb += b_st["fallbacks"]

    mean_score = sum(game_scores) / max(1, len(records))
    elo_gain = compute_elo_difference(mean_score)

    rng = np.random.default_rng(0)
    arr = np.array(game_scores)
    boot = arr[rng.integers(0, len(arr), size=(500, len(arr)))].mean(axis=1)
    elo_ci95 = [
        round(compute_elo_difference(float(np.percentile(boot, 2.5))), 1),
        round(compute_elo_difference(float(np.percentile(boot, 97.5))), 1),
    ]

    c_n = max(1, cand_dec)
    b_n = max(1, base_dec)

    return {
        "games": len(records),
        "n_openings": n_openings,
        "seed": seed,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": round(mean_score, 4),
        "elo_gain": round(elo_gain, 1),
        "elo_gain_ci95": elo_ci95,
        "candidate_legality": {
            "decisions": cand_dec,
            "legal_first_try": round(cand_a1 / c_n, 4),
            "legal_within_3": round(cand_a123 / c_n, 4),
            "random_fallback_rate": round(cand_fb / c_n, 4),
        },
        "base_legality": {
            "decisions": base_dec,
            "legal_first_try": round(base_a1 / b_n, 4),
            "legal_within_3": round(base_a123 / b_n, 4),
            "random_fallback_rate": round(base_fb / b_n, 4),
        },
    }


def resolve_default_candidate():
    for p in (
        POST_TRAINING_DIR / "checkpoints" / "rl_best" / "model.safetensors",
        POST_TRAINING_DIR / "checkpoints" / "sft_best" / "model.safetensors",
        POST_TRAINING_DIR / "checkpoints" / "rl_latest" / "model.safetensors",
        POST_TRAINING_DIR / "checkpoints" / "sft_latest" / "model.safetensors",
        POST_TRAINING_DIR / "final_model" / "model.safetensors",
    ):
        if p.exists():
            return p
    return REPO_ROOT / "base_model" / "model.safetensors"


def export_and_verify(args):
    src_path = Path(args.checkpoint) if args.checkpoint else resolve_default_candidate()
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    model = load_any_checkpoint(src_path)
    exported_file = save_model(model, export_dir)

    sf_errors = check_safetensors_file(exported_file)
    if sf_errors:
        raise RuntimeError(f"safetensors check failed: {sf_errors}")

    base_file = REPO_ROOT / "base_model" / "model.safetensors"
    base_model = load_model(base_file, device="cpu")

    cand_sha = sha256_file(exported_file)
    base_sha = sha256_file(base_file)
    is_modified = cand_sha != base_sha

    cand_sd = model.state_dict()
    base_sd = base_model.state_dict()
    total_l2_sq = 0.0
    max_abs_diff = 0.0
    per_tensor_l2 = {}
    for k in cand_sd:
        diff = cand_sd[k].float() - base_sd[k].float()
        l2 = float(torch.norm(diff, p=2).item())
        per_tensor_l2[k] = round(l2, 5)
        total_l2_sq += l2 * l2
        max_abs_diff = max(max_abs_diff, float(diff.abs().max().item()))
    total_l2 = math.sqrt(total_l2_sq)

    print(f"[export] src={src_path} -> {exported_file} ({exported_file.stat().st_size:,} bytes)")
    print(f"[export] modified={is_modified} | l2_delta={total_l2:.4f} | max_delta={max_abs_diff:.5f}")

    report = {
        "source_checkpoint": str(src_path),
        "exported_safetensors": str(exported_file),
        "sha256": cand_sha,
        "base_sha256": base_sha,
        "weights_modified_from_base": is_modified,
        "total_l2_delta": round(total_l2, 4),
        "max_abs_delta": round(max_abs_diff, 6),
        "per_tensor_l2_delta": per_tensor_l2,
    }

    if not args.skip_match:
        h2h = evaluate_head_to_head(exported_file, base_file, n_openings=args.n_openings, seed=args.seed, workers=args.workers)
        report["head_to_head_vs_base"] = h2h
        print(f"[export] h2h vs base: W={h2h['wins']} D={h2h['draws']} L={h2h['losses']} score={h2h['score']:.3f} ({h2h['elo_gain']:+.1f} Elo)")

    if args.run_engine_eval:
        eng_eval = evaluate_vs_engine(str(exported_file), levels=args.levels, n_openings=args.n_openings, seed=0, workers=args.workers)
        report["engine_evaluation"] = eng_eval
        print(f"[export] engine L{args.levels}: mean_score={eng_eval['mean_score']:.3f} top1_leg={eng_eval['legal_first_try']:.2%} top3_leg={eng_eval['legal_within_3']:.2%}")

    writeup_path = Path(args.writeup)
    if writeup_path.exists() and is_modified:
        cmd = [sys.executable, str(REPO_ROOT / "submit.py"), "--model", str(export_dir), "--writeup", str(writeup_path)]
        if not args.upload:
            cmd.append("--dry_run")
        proc = subprocess.run(cmd, text=True, capture_output=True)
        print(proc.stdout.strip())
        if proc.returncode != 0:
            print(f"[export] submit.py error:\n{proc.stderr.strip()}")
        report["submission_check_passed"] = (proc.returncode == 0)

    out_json = export_dir / "verification_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[export] report saved to {out_json}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--export_dir", default=str(POST_TRAINING_DIR / "final_model"))
    ap.add_argument("--writeup", default=str(POST_TRAINING_DIR / "WRITEUP.md"))
    ap.add_argument("--n_openings", type=int, default=25)
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--skip_match", action="store_true")
    ap.add_argument("--run_engine_eval", action="store_true")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    export_and_verify(args)


if __name__ == "__main__":
    main()
