#!/usr/bin/env python3
"""Example post-training: make the model put its probability mass on LEGAL moves.

This is a deliberately simple baseline, to show the mechanics:
  1. collect positions by letting the current model play against the engine,
  2. minimise  -log sum_{m legal} p_theta(m | history)      (legality loss)
             + beta * KL(p_base || p_theta) on legal moves  (stay close to the base policy)

Using python-chess / engines / Stockfish at TRAINING time is allowed: only
the inference procedure (chess_hw/play.py) is fixed.

    uv run examples/legal_finetune.py --out my_model --rounds 3
"""

import argparse
import random
import sys
from pathlib import Path

import chess
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chess_hw import ChessTokenizer, load_model, save_model  # noqa: E402
from chess_hw.engine import Engine  # noqa: E402
from chess_hw.match import load_openings  # noqa: E402
from chess_hw.play import ModelPlayer  # noqa: E402

TOK = ChessTokenizer()


def collect_positions(model, n_games, level, rng, max_plies=200):
    """Play model vs engine; when the model fails, play a random legal move instead and continue."""
    player = ModelPlayer(model)
    book = load_openings()
    data = []
    for g in range(n_games):
        board = chess.Board()
        for u in rng.choice(book):
            board.push_uci(u)
        engine = Engine(level, seed=g)
        model_color = rng.random() < 0.5
        while not board.is_game_over() and board.ply() < max_plies:
            if board.turn == model_color:
                hist = [m.uci() for m in board.move_stack]
                legal = [TOK.token_to_id[board.uci(m)] for m in board.legal_moves]
                data.append((TOK.encode(hist), legal))
                mv = player.choose(board).move or rng.choice(list(board.legal_moves))
            else:
                mv = engine.best_move(board)
            board.push(mv)
    return data


def batches(data, bs, rng):
    idx = list(range(len(data)))
    rng.shuffle(idx)
    for s in range(0, len(idx), bs):
        yield [data[i] for i in idx[s:s + bs]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(Path(__file__).resolve().parents[1] / "base_model"))
    ap.add_argument("--out", default="my_model")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--games", type=int, default=200, help="games collected per round")
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = load_model(args.base, device=dev)
    model = load_model(args.base, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    for r in range(args.rounds):
        model.eval()
        data = collect_positions(model, args.games, args.level, rng)
        print(f"round {r}: {len(data)} positions")
        model.train()
        for ep in range(args.epochs):
            tot, n = 0.0, 0
            for batch in batches(data, args.bs, rng):
                L = max(len(x) for x, _ in batch)
                x = torch.zeros(len(batch), L, dtype=torch.long)
                mask = torch.zeros(len(batch), TOK.vocab_size, dtype=torch.bool)
                for i, (ids, legal) in enumerate(batch):
                    x[i, :len(ids)] = torch.tensor(ids)
                    mask[i, legal] = True
                x, mask = x.to(dev), mask.to(dev)
                last = torch.tensor([len(ids) - 1 for ids, _ in batch], device=dev)
                rows = torch.arange(len(batch), device=dev)
                logp = F.log_softmax(model(x)[rows, last], dim=-1)
                with torch.no_grad():
                    logp_base = F.log_softmax(base(x)[rows, last], dim=-1)
                legal_mass = torch.logsumexp(logp.masked_fill(~mask, -1e9), dim=-1)
                # KL between base and model, both renormalised over legal moves
                q_base = F.softmax(logp_base.masked_fill(~mask, -1e9), dim=-1)
                logq = (logp - legal_mass[:, None]).masked_fill(~mask, 0.0)
                kl = (q_base * (torch.log(q_base.clamp_min(1e-12)) - logq)).masked_fill(~mask, 0.0).sum(-1)
                loss = (-legal_mass + args.beta * kl).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += loss.item() * len(batch)
                n += len(batch)
            print(f"  epoch {ep}: loss {tot / n:.4f}")
    model.eval()
    print("saved", save_model(model.cpu(), args.out))


if __name__ == "__main__":
    main()
