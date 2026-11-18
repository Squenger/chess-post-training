#!/usr/bin/env python3
"""Evaluate a model against the deterministic engine.

    uv run evaluate.py --model my_model/            # a dir with model.safetensors
    uv run evaluate.py --model base_model/ --levels 0 1 2 --n_openings 20 --workers 8

This is the same code as the official evaluation. The official run uses
different (private) seeds, so do not over-fit to seed 0.
"""

import argparse
import json
from pathlib import Path

from chess_hw.match import evaluate_vs_engine


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="base_model", help="directory containing model.safetensors (or the file)")
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--n_openings", type=int, default=50, help="each opening is played with both colors")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None, help="write summary JSON here (games go to <out>.games.jsonl)")
    args = ap.parse_args()

    games_out = f"{args.out}.games.jsonl" if args.out else None
    s = evaluate_vs_engine(args.model, args.levels, args.n_openings, args.seed, args.workers, games_out=games_out)

    print(f"\nModel: {s['model']}   seed={s['seed']}   openings={s['n_openings']} (x2 colors)")
    print(f"Legal first try: {s['legal_first_try']:.1%}   legal within 3: {s['legal_within_3']:.2%}   "
          f"random moves (3 illegal attempts): {s['random_move_rate']:.2%}")
    print(f"{'level':>5} {'games':>6} {'W':>4} {'D':>4} {'L':>4} {'rand':>5} {'score':>6}")
    for lvl, d in s["per_level"].items():
        print(f"{lvl:>5} {d['games']:>6} {d['wins']:>4} {d['draws']:>4} {d['losses']:>4} {d['random_moves']:>5} {d['score']:>6.3f}")
    print(f"Mean score over levels: {s['mean_score']:.3f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
