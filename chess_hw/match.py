"""Game runner, opening book and evaluation against the engine.

Everything is deterministic given a seed:
  * the seed selects which openings (from the bundled opening book) are used,
  * the model plays greedily (see play.py),
  * the engine is deterministic (level 0 uses an RNG derived from the seed).
Each opening is played twice: once with the model as White, once as Black.

DO NOT MODIFY: the evaluation uses this exact file (with private seeds).
"""

from __future__ import annotations

import gzip
import json
import random
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import chess
import torch

from .engine import Engine
from .model import check_not_lfs_pointer, load_model
from .play import ModelPlayer, MoveChoice

PACKAGE_DIR = Path(__file__).resolve().parent
OPENINGS_FILE = PACKAGE_DIR / "openings.txt.gz"
MAX_PLIES = 250  # game is adjudicated a draw after this many plies (fits in n_ctx=256)
# When none of the model's 3 attempts is legal, a random legal move is played instead
# (seeded by opening and ply, hence reproducible) and the game goes on.


# ---------------------------------------------------------------------------
# Openings
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_openings() -> List[List[str]]:
    check_not_lfs_pointer(OPENINGS_FILE)
    with gzip.open(OPENINGS_FILE, "rt") as f:
        return [line.split() for line in f if line.strip()]


def select_openings(seed: int, n: int) -> List[int]:
    """Indices of n distinct openings, determined by the seed."""
    book = load_openings()
    return random.Random(f"openings-{seed}").sample(range(len(book)), n)


# ---------------------------------------------------------------------------
# Players and games
# ---------------------------------------------------------------------------

class EnginePlayer:
    def __init__(self, level: int, seed: str = "0"):
        self.name = f"engine-L{level}"
        self.engine = Engine(level=level, seed=seed)

    def choose(self, board: chess.Board) -> MoveChoice:
        return MoveChoice(self.engine.best_move(board), 1)


@dataclass
class SideStats:
    decisions: int = 0  # number of times this side had to move
    attempts: List[int] = field(default_factory=lambda: [0, 0, 0])  # legal at attempt 1/2/3
    fallbacks: int = 0  # random moves played for this side because all 3 attempts were illegal


@dataclass
class GameRecord:
    white: str
    black: str
    opening_id: int
    moves: List[str]
    result: str  # "1-0", "0-1", "1/2-1/2"
    termination: str
    white_stats: SideStats
    black_stats: SideStats
    meta: Dict = field(default_factory=dict)


def play_game(white, black, opening: Sequence[str], opening_id: int = -1, max_plies: int = MAX_PLIES) -> GameRecord:
    board = chess.Board()
    for u in opening:
        board.push_uci(u)
    stats = {chess.WHITE: SideStats(), chess.BLACK: SideStats()}
    result, termination = None, None
    while result is None:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            result, termination = outcome.result(), outcome.termination.name.lower()
            break
        if board.ply() >= max_plies:
            result, termination = "1/2-1/2", "max_plies"
            break
        side = board.turn
        player = white if side == chess.WHITE else black
        st = stats[side]
        st.decisions += 1
        choice = player.choose(board)
        if choice.move is None:
            st.fallbacks += 1
            legal = sorted(board.legal_moves, key=lambda m: m.uci())
            board.push(random.Random(f"fallback-{opening_id}-{board.ply()}").choice(legal))
            continue
        st.attempts[choice.attempt - 1] += 1
        board.push(choice.move)
    return GameRecord(
        white=white.name, black=black.name, opening_id=opening_id,
        moves=[m.uci() for m in board.move_stack], result=result, termination=termination,
        white_stats=stats[chess.WHITE], black_stats=stats[chess.BLACK],
    )


def score_for(result: str, color: chess.Color) -> float:
    if result == "1/2-1/2":
        return 0.5
    white_won = result == "1-0"
    return 1.0 if white_won == (color == chess.WHITE) else 0.0


# ---------------------------------------------------------------------------
# Evaluation against the engine (multiprocessing over games, CPU, 1 thread each)
# ---------------------------------------------------------------------------

_WORKER: Dict = {}


def _init_worker(model_paths: Dict[str, str]):
    torch.set_num_threads(1)
    _WORKER["paths"] = model_paths
    _WORKER["models"] = {}


def _get_player(name: str) -> ModelPlayer:
    models = _WORKER["models"]
    if name not in models:
        models[name] = ModelPlayer(load_model(_WORKER["paths"][name], device="cpu"), name=name)
    return models[name]


def _engine_task(task):
    name, level, seed, oid, model_color = task
    model = _get_player(name)
    engine = EnginePlayer(level, seed=f"engine-{seed}-{level}-{oid}-{model_color}")
    opening = load_openings()[oid]
    white, black = (model, engine) if model_color == "white" else (engine, model)
    rec = play_game(white, black, opening, oid)
    rec.meta = {"level": level, "seed": seed, "model_color": model_color}
    return asdict(rec)


def _pair_task(task):
    """Model vs model game: task = (name_a, name_b, opening_id, a_is_white)."""
    a, b, oid, a_white = task
    pa, pb = _get_player(a), _get_player(b)
    white, black = (pa, pb) if a_white else (pb, pa)
    rec = play_game(white, black, load_openings()[oid], oid)
    rec.meta = {"a": a, "b": b, "a_white": a_white}
    return asdict(rec)


def run_tasks(fn, tasks, model_paths: Dict[str, str], workers: int, progress: bool = True, on_result=None):
    """Run fn(task) for every task in a process pool; on_result(i, rec) is called as results arrive."""
    ctx = get_context("spawn")
    out = [None] * len(tasks)
    with ctx.Pool(workers, initializer=_init_worker, initargs=(model_paths,)) as pool:
        it = pool.imap_unordered(_indexed, [(fn, i, t) for i, t in enumerate(tasks)], chunksize=1)
        try:
            from tqdm import tqdm
            it = tqdm(it, total=len(tasks), disable=not progress, desc="games")
        except ImportError:
            pass
        for i, rec in it:
            out[i] = rec
            if on_result is not None:
                on_result(i, rec)
    return out


def _indexed(args):
    fn, i, t = args
    return i, fn(t)


def summarize_vs_engine(records: List[dict]) -> dict:
    per_level: Dict[int, dict] = {}
    tot = {"decisions": 0, "a1": 0, "a2": 0, "a3": 0, "fallbacks": 0}
    for r in records:
        lvl = r["meta"]["level"]
        color = chess.WHITE if r["meta"]["model_color"] == "white" else chess.BLACK
        st = r["white_stats"] if color == chess.WHITE else r["black_stats"]
        d = per_level.setdefault(lvl, {"games": 0, "wins": 0, "draws": 0, "losses": 0, "random_moves": 0, "score": 0.0})
        s = score_for(r["result"], color)
        d["games"] += 1
        d["wins"] += s == 1.0
        d["draws"] += s == 0.5
        d["losses"] += s == 0.0
        d["random_moves"] += st["fallbacks"]
        d["score"] += s
        tot["decisions"] += st["decisions"]
        tot["fallbacks"] += st["fallbacks"]
        for k in range(3):
            tot[f"a{k + 1}"] += st["attempts"][k]
    for d in per_level.values():
        d["score"] = d["score"] / d["games"]
    n = max(tot["decisions"], 1)
    return {
        "per_level": {str(k): v for k, v in sorted(per_level.items())},
        "mean_score": sum(d["score"] for d in per_level.values()) / max(len(per_level), 1),
        "legal_first_try": tot["a1"] / n,
        "legal_within_3": (tot["a1"] + tot["a2"] + tot["a3"]) / n,
        "model_moves": tot["decisions"],
        "random_move_rate": tot["fallbacks"] / n,  # fraction of model moves replaced by a random move
    }


def evaluate_vs_engine(
    model_path: str,
    levels: Sequence[int] = (0, 1, 2, 3),
    n_openings: int = 50,
    seed: int = 0,
    workers: int = 4,
    progress: bool = True,
    games_out: Optional[str] = None,
) -> dict:
    oids = select_openings(seed, n_openings)
    tasks = [("m", lvl, seed, oid, color) for lvl in levels for oid in oids for color in ("white", "black")]
    records = run_tasks(_engine_task, tasks, {"m": str(model_path)}, workers, progress)
    if games_out:
        Path(games_out).parent.mkdir(parents=True, exist_ok=True)
        with open(games_out, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    summary = summarize_vs_engine(records)
    summary.update({"model": str(model_path), "seed": seed, "n_openings": n_openings, "levels": list(levels)})
    return summary
