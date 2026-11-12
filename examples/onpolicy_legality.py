#!/usr/bin/env python3
"""On-policy legality training (+ optional engine distillation and Lichess replay).

A stronger baseline than legal_finetune.py. Used by the instructor to check that
post-training helps: 4 rounds x 3000 games (~25 min on a 20-core machine + GPU) took
the mean score at seed 0 (levels 0-3) from 0.225 to 0.256.

Each round:
  1. play the current model against engine levels 1-2 from book openings (random legal
     fallback when it fails, so games go on); optionally label the model's positions with
     the engine at --teacher level;
  2. train on whole games with
       legality loss  -log P(legal) at EVERY ply (both sides)
     + lam_ce  * CE(teacher move) at the model's plies (distillation, if --teacher > 0)
     + lam_rep * next-move CE on Lichess games (replay, against forgetting).
Data is aggregated over rounds.

    uv run examples/onpolicy_legality.py --out my_model_rounds --rounds 4 --games 3000
    # optional replay data, built with the same conversion as the base model (UCI token arrays):
    #   --data path/to/train   (expects train.bin / train.idx, see README)
"""

import argparse
import random
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chess_hw import ChessTokenizer, load_model, save_model  # noqa: E402
from chess_hw.engine import Engine  # noqa: E402
from chess_hw.match import load_openings  # noqa: E402
from chess_hw.play import ModelPlayer  # noqa: E402

TOK = ChessTokenizer()
_M = {}


def _init(path):
    torch.set_num_threads(1)
    _M["p"] = ModelPlayer(load_model(path))


def play_game(task):
    """Returns (token ids, per-ply legal ids, per-ply teacher id or -1)."""
    gi, seed, teacher_level = task
    rng = random.Random(f"{seed}-{gi}")
    board = chess.Board()
    for u in rng.choice(load_openings()):
        board.push_uci(u)
    opp = Engine(rng.choice([1, 2]))
    teacher = Engine(teacher_level) if teacher_level > 0 else None
    color = rng.random() < 0.5
    while not board.is_game_over(claim_draw=True) and board.ply() < 200:
        if board.turn == color:
            mv = _M["p"].choose(board).move or rng.choice(sorted(board.legal_moves, key=lambda m: m.uci()))
        else:
            mv = opp.best_move(board)
        board.push(mv)
    # replay to build per-ply targets
    moves = [m.uci() for m in board.move_stack]
    b = chess.Board()
    legal, teach = [], []
    for u in moves:
        legal.append(np.array([TOK.token_to_id[b.uci(m)] for m in b.legal_moves], dtype=np.int16))
        if teacher is not None and b.turn == color and b.ply() >= 8:
            teach.append(TOK.token_to_id[b.uci(teacher.best_move(b))])
        else:
            teach.append(-1)
        b.push_uci(u)
    return np.array(TOK.encode(moves), dtype=np.int64), legal, np.array(teach, dtype=np.int64)


def collect(model_path, n_games, seed, workers, teacher_level):
    with get_context("spawn").Pool(workers, initializer=_init, initargs=(str(model_path),)) as pool:
        return pool.map(play_game, [(g, seed, teacher_level) for g in range(n_games)], chunksize=4)


class Lichess:
    def __init__(self, prefix):
        self.ids = np.fromfile(f"{prefix}.bin", dtype=np.uint16)
        self.off = np.fromfile(f"{prefix}.idx", dtype=np.int64)

    def batch(self, bs, rng):
        idx = rng.integers(0, len(self.off) - 1, bs)
        gs = [self.ids[self.off[i]:self.off[i + 1]].astype(np.int64) for i in idx]
        x = np.zeros((bs, max(len(g) for g in gs)), dtype=np.int64)
        for r, g in enumerate(gs):
            x[r, :len(g)] = g
        return torch.from_numpy(x)


def game_batch(games, dev):
    T = max(len(g[0]) for g in games) - 1  # predict tokens 1..n from prefixes 0..n-1
    B = len(games)
    x = torch.zeros(B, T, dtype=torch.long)
    mask = torch.zeros(B, T, TOK.vocab_size, dtype=torch.bool)
    valid = torch.zeros(B, T, dtype=torch.bool)
    teach = torch.full((B, T), -100, dtype=torch.long)
    for i, (ids, legal, tch) in enumerate(games):
        n = len(ids) - 1
        x[i, :n] = torch.from_numpy(ids[:-1])
        valid[i, :n] = True
        for t in range(n):
            mask[i, t, torch.from_numpy(legal[t].astype(np.int64))] = True
        tt = torch.from_numpy(tch)
        teach[i, :n] = torch.where(tt >= 0, tt, torch.full_like(tt, -100))
    return x.to(dev), mask.to(dev), valid.to(dev), teach.to(dev)


def train(model, games, lich, args, dev, rng):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    model.train()
    order = list(range(len(games)))
    for ep in range(args.epochs):
        random.shuffle(order)
        stats = np.zeros(3)
        nb = 0
        for s in range(0, len(order), args.bs):
            x, mask, valid, teach = game_batch([games[i] for i in order[s:s + args.bs]], dev)
            logits = model(x)
            logp = F.log_softmax(logits.float(), -1)
            legal_mass = torch.logsumexp(logp.masked_fill(~mask, -1e9), -1)
            l_legal = -(legal_mass * valid).sum() / valid.sum()
            l_ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), teach.reshape(-1), ignore_index=-100) \
                if (teach >= 0).any() else torch.zeros((), device=dev)
            if lich is not None:
                xr = lich.batch(args.bs, rng).to(dev)
                lr_logits = model(xr[:, :-1])
                yr = xr[:, 1:].clone()
                yr[yr == 0] = -100
                l_rep = F.cross_entropy(lr_logits.reshape(-1, lr_logits.size(-1)).float(), yr.reshape(-1), ignore_index=-100)
            else:
                l_rep = torch.zeros((), device=dev)
            loss = l_legal + args.lam_ce * l_ce + args.lam_rep * l_rep
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            stats += [l_legal.item(), l_ce.item(), l_rep.item()]
            nb += 1
        print(f"  epoch {ep}: legal {stats[0] / nb:.4f} ce {stats[1] / nb:.4f} replay {stats[2] / nb:.4f}", flush=True)
    model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "base_model"))
    ap.add_argument("--data", default=None, help="prefix of .bin/.idx Lichess token arrays for replay (optional)")
    ap.add_argument("--out", default="work/pt")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--teacher", type=int, default=0, help="engine level used as teacher (0 = no distillation)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lam_ce", type=float, default=0.5)
    ap.add_argument("--lam_rep", type=float, default=0.5)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    lich = Lichess(args.data) if args.data else None
    out = Path(args.out)
    model = load_model(args.base, device=dev)
    cur = Path(args.base)
    games = []
    for r in range(args.rounds):
        t = time.time()
        new = collect(cur, args.games, seed=r, workers=args.workers, teacher_level=args.teacher)
        games += new
        import pickle
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"games_round{r}.pkl", "wb") as fh:
            pickle.dump(new, fh)
        print(f"round {r}: +{len(new)} games, {sum(len(g[0]) for g in new)} plies, total {len(games)} games "
              f"({time.time() - t:.0f}s)", flush=True)
        t = time.time()
        train(model, games, lich, args, dev, rng)
        cur = save_model(model, out / f"round{r}").parent
        print(f"  trained in {time.time() - t:.0f}s -> {cur}", flush=True)


if __name__ == "__main__":
    main()
