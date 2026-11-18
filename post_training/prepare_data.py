#!/usr/bin/env python3
"""Build SFT training sequences and Stockfish-annotated validation positions."""

import argparse
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
from tqdm import tqdm

POST_TRAINING_DIR = Path(__file__).resolve().parent
REPO_ROOT = POST_TRAINING_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(POST_TRAINING_DIR))

from chess_hw import CONFIG, ChessTokenizer, Engine, ModelPlayer, load_model
from chess_hw.engine import MATE
from chess_hw.match import MAX_PLIES, load_openings
from inspect_repo import resolve_stockfish_path

TOK = ChessTokenizer()
_WORKER = {}


def _init_data_worker(stockfish_path, sf_depth, sf_time_limit, base_model_path, teacher_level):
    torch.set_num_threads(1)
    _WORKER["tok"] = ChessTokenizer()
    _WORKER["sf_depth"] = sf_depth
    _WORKER["sf_time"] = sf_time_limit
    _WORKER["teacher"] = Engine(level=teacher_level, seed=42)
    _WORKER["sf_engine"] = None

    if stockfish_path and Path(stockfish_path).is_file():
        try:
            eng = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            eng.configure({"Threads": 1, "Hash": 16})
            _WORKER["sf_engine"] = eng
        except Exception:
            _WORKER["sf_engine"] = None

    if base_model_path and Path(base_model_path).exists():
        _WORKER["player"] = ModelPlayer(load_model(base_model_path, device="cpu"), name="base")
    else:
        _WORKER["player"] = None


def _query_teacher(board, top_k_evals=1):
    sf_eng = _WORKER.get("sf_engine")
    if sf_eng is not None:
        try:
            limit = chess.engine.Limit(
                depth=_WORKER["sf_depth"],
                time=_WORKER["sf_time"] if _WORKER["sf_time"] > 0 else None,
            )
            multipv = max(1, min(top_k_evals, board.legal_moves.count()))
            infos = sf_eng.analyse(board, limit, multipv=multipv)
            if isinstance(infos, dict):
                infos = [infos]
            evals_cp = {}
            best_mv = None
            best_cp = -10000
            for info in infos:
                pv = info.get("pv")
                if not pv:
                    continue
                mv = pv[0]
                cp = info["score"].pov(board.turn).score(mate_score=10000)
                cp = 0 if cp is None else int(cp)
                evals_cp[board.uci(mv)] = cp
                if best_mv is None:
                    best_mv = mv
                    best_cp = cp
            if best_mv is not None:
                return best_mv, best_cp, evals_cp
        except Exception:
            pass

    teacher = _WORKER["teacher"]
    legal_moves = list(board.legal_moves)
    best_mv = legal_moves[0]
    best_cp = -MATE - 1
    evals_cp = {}
    b_copy = board.copy(stack=False)
    for mv in legal_moves:
        b_copy.push(mv)
        score = -teacher._search(b_copy, max(0, teacher.level - 1), -MATE - 1, MATE + 1, 1)
        b_copy.pop()
        evals_cp[board.uci(mv)] = int(score)
        if score > best_cp:
            best_cp = int(score)
            best_mv = mv
    return best_mv, best_cp, evals_cp


def _is_tactical(board, move):
    if board.is_check() or board.is_capture(move) or move.promotion is not None:
        return True
    board.push(move)
    gives_check = board.is_check()
    board.pop()
    return gives_check


def _encode_annotated_game(moves_uci, pillar, relabel_tactical=False, focus_color=None, tactical_stride=2):
    tok = _WORKER.get("tok", TOK)
    board = chess.Board()
    valid_moves = []

    for u in moves_uci[: MAX_PLIES - 1]:
        try:
            mv = board.parse_uci(u)
        except ValueError:
            break
        valid_moves.append(board.uci(mv))
        board.push(mv)
        if board.is_game_over(claim_draw=True):
            break

    if len(valid_moves) < 8:
        return None

    board = chess.Board()
    input_ids = [tok.bos_token_id] + [tok.token_to_id[u] for u in valid_moves[:-1]]
    L = len(input_ids)
    target_ids = np.full(L, -100, dtype=np.int16)
    weights = np.ones(L, dtype=np.float16)
    legal_lists = []

    for t, u in enumerate(valid_moves):
        legal_ids = np.array([tok.token_to_id[board.uci(m)] for m in board.legal_moves], dtype=np.uint16)
        legal_lists.append(legal_ids)
        played_mv = chess.Move.from_uci(u)
        game_tok_id = tok.token_to_id[u]
        is_tact = _is_tactical(board, played_mv)

        if pillar == 0:
            if focus_color is None or board.turn == focus_color:
                target_ids[t] = game_tok_id
                weights[t] = 1.25 if is_tact else 1.0
            else:
                target_ids[t] = game_tok_id
                weights[t] = 0.75
        elif pillar == 1:
            should_query = relabel_tactical and t >= 6 and (is_tact or (t % tactical_stride == 0))
            if should_query and (focus_color is None or board.turn == focus_color):
                best_mv, _, _ = _query_teacher(board, top_k_evals=1)
                best_uci = board.uci(best_mv)
                target_ids[t] = tok.token_to_id[best_uci]
                weights[t] = 2.5 if (is_tact or best_uci != u) else 1.75
            else:
                target_ids[t] = game_tok_id
                weights[t] = 1.25 if is_tact else 0.8
        else:
            if relabel_tactical and (board.is_check() or len(legal_ids) <= 12) and t >= 8:
                best_mv, _, _ = _query_teacher(board, top_k_evals=1)
                target_ids[t] = tok.token_to_id[board.uci(best_mv)]
            else:
                target_ids[t] = game_tok_id
            weights[t] = 1.5 if (t >= 50 or board.is_check()) else 1.0

        board.push(played_mv)

    # Store ragged legal move lists in CSR layout to keep the saved .pt compact
    legal_offsets = np.zeros(L + 1, dtype=np.int32)
    curr = 0
    for idx, arr in enumerate(legal_lists):
        curr += len(arr)
        legal_offsets[idx + 1] = curr
    legal_packed = np.concatenate(legal_lists) if legal_lists else np.zeros(0, dtype=np.uint16)

    return {
        "input_ids": np.array(input_ids, dtype=np.uint16),
        "target_ids": target_ids,
        "weights": weights,
        "legal_packed": legal_packed,
        "legal_offsets": legal_offsets,
        "pillar": np.uint8(pillar),
    }


def _worker_process_task(task):
    try:
        task_type, payload = task

        if task_type == "hf_game":
            moves_uci, pillar, winner_color, relabel = payload
            return _encode_annotated_game(moves_uci, pillar, relabel_tactical=relabel,
                                          focus_color=winner_color, tactical_stride=3)

        if task_type == "pv_game":
            opening_moves, seed, pillar = payload
            rng = random.Random(seed)
            board = chess.Board()
            for u in opening_moves:
                board.push_uci(u)
            opp = Engine(level=rng.choice([1, 2]), seed=seed)
            teacher_color = chess.WHITE if (seed % 2 == 0) else chess.BLACK

            while not board.is_game_over(claim_draw=True) and board.ply() < 120:
                if board.turn == teacher_color or rng.random() < 0.5:
                    mv, _, _ = _query_teacher(board, top_k_evals=1)
                else:
                    mv = opp.best_move(board)
                if mv is None:
                    break
                board.push(mv)

            moves_uci = [m.uci() for m in board.move_stack]
            return _encode_annotated_game(moves_uci, pillar, relabel_tactical=(pillar == 1),
                                          focus_color=teacher_color, tactical_stride=2)

        if task_type == "onpolicy_tactical":
            opening_moves, seed, pillar = payload
            rng = random.Random(seed)
            board = chess.Board()
            for u in opening_moves:
                board.push_uci(u)
            opp = Engine(level=rng.choice([1, 2]), seed=seed)
            player = _WORKER.get("player")
            model_color = chess.WHITE if (seed % 2 == 0) else chess.BLACK

            while not board.is_game_over(claim_draw=True) and board.ply() < 140:
                if board.turn == model_color and player is not None:
                    mv = player.choose(board).move
                    if mv is None:
                        mv = rng.choice(sorted(board.legal_moves, key=lambda m: m.uci()))
                else:
                    mv = opp.best_move(board)
                if mv is None:
                    break
                board.push(mv)

            moves_uci = [m.uci() for m in board.move_stack]
            return _encode_annotated_game(moves_uci, pillar, relabel_tactical=True,
                                          focus_color=model_color if pillar == 1 else None,
                                          tactical_stride=2 if pillar == 1 else 3)
    except Exception as e:
        print(f"[prepare_data worker] {e}", flush=True)

    return None


def _worker_build_val_position(task):
    idx, moves_uci, target_ply = task
    tok = _WORKER.get("tok", TOK)
    board = chess.Board()
    history = []

    for ply_i, u in enumerate(moves_uci):
        if ply_i >= target_ply or board.is_game_over(claim_draw=True):
            break
        try:
            mv = board.parse_uci(u)
        except ValueError:
            break
        history.append(board.uci(mv))
        board.push(mv)

    if board.is_game_over(claim_draw=True) or board.legal_moves.count() == 0 or not history:
        return None
    if len(history) >= CONFIG.n_ctx:
        history = history[: CONFIG.n_ctx - 1]

    legal_moves = list(board.legal_moves)
    legal_ucis = [board.uci(m) for m in legal_moves]
    legal_ids = [tok.token_to_id[u] for u in legal_ucis]

    best_mv, best_cp, evals_cp = _query_teacher(board, top_k_evals=min(12, len(legal_moves)))
    best_uci = board.uci(best_mv)

    ply = board.ply()
    if board.is_check():
        phase = "tactical_check"
    elif ply <= 16:
        phase = "opening"
    elif ply <= 70:
        phase = "middlegame"
    else:
        phase = "endgame"

    return {
        "id": idx,
        "history": history,
        "input_ids": tok.encode(history),
        "fen": board.fen(),
        "ply": ply,
        "phase": phase,
        "legal_ucis": legal_ucis,
        "legal_ids": legal_ids,
        "best_move_uci": best_uci,
        "best_move_id": tok.token_to_id[best_uci],
        "best_cp": int(best_cp),
        "move_evals_cp": evals_cp,
    }


def fetch_hf_games(max_games, seed=42):
    games_out = []
    if max_games <= 0:
        return games_out

    try:
        from datasets import load_dataset
        print("[prepare_data] streaming dlouapre/lichess_2025-01_1M...")
        ds = load_dataset("dlouapre/lichess_2025-01_1M", split="train", streaming=True)
        for row in ds:
            text = row.get("text") or row.get("moves") or row.get("uci") or ""
            if not text:
                continue
            moves = [m for m in text.strip().split()
                     if m in TOK.token_to_id and TOK.token_to_id[m] >= TOK.n_special]
            if 24 <= len(moves) <= MAX_PLIES:
                games_out.append((moves, None))
                if len(games_out) >= max_games:
                    break
        print(f"[prepare_data] loaded {len(games_out)} HF games")
    except Exception as e:
        print(f"[prepare_data] HF stream unavailable ({e}), using local openings + Stockfish")

    return games_out


def build_sft_dataset(
    out_dir,
    total_sequences=2000,
    val_positions_count=1000,
    workers=8,
    sf_depth=12,
    sf_time_limit=0.04,
    teacher_level=2,
    use_hf_dataset=True,
    seed=42,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    openings = load_openings()
    sf_path = resolve_stockfish_path(auto_download=False)
    base_model_path = str(REPO_ROOT / "base_model")

    n_p1 = int(round(total_sequences * 0.40))
    n_p2 = int(round(total_sequences * 0.40))
    n_p3 = total_sequences - n_p1 - n_p2

    print(f"[prepare_data] n_seq={total_sequences} (mix={n_p1}/{n_p2}/{n_p3}), n_val={val_positions_count}, workers={workers}")
    print(f"[prepare_data] engine={sf_path or f'Engine L{teacher_level}'}")

    hf_games = fetch_hf_games(n_p1 + n_p3, seed=seed) if use_hf_dataset else []
    tasks = []

    for i in range(n_p1):
        if i < len(hf_games) // 2:
            moves_uci, win_col = hf_games[i]
            tasks.append(("hf_game", (moves_uci, 0, win_col, False)))
        else:
            op = openings[(seed + i * 13) % len(openings)]
            tasks.append(("pv_game", (op, seed + i, 0)))

    for i in range(n_p2):
        op = openings[(seed + 10000 + i * 17) % len(openings)]
        tasks.append(("onpolicy_tactical", (op, seed + 10000 + i, 1)))

    for i in range(n_p3):
        hf_idx = len(hf_games) // 2 + i
        if hf_idx < len(hf_games):
            moves_uci, win_col = hf_games[hf_idx]
            tasks.append(("hf_game", (moves_uci, 2, win_col, True)))
        else:
            op = openings[(seed + 20000 + i * 23) % len(openings)]
            tasks.append(("onpolicy_tactical", (op, seed + 20000 + i, 2)))

    t0 = time.time()
    samples = []
    ctx = get_context("spawn")
    with ctx.Pool(
        processes=workers,
        initializer=_init_data_worker,
        initargs=(sf_path, sf_depth, sf_time_limit, base_model_path, teacher_level),
    ) as pool:
        for res in tqdm(
            pool.imap_unordered(_worker_process_task, tasks, chunksize=max(1, len(tasks) // (workers * 4))),
            total=len(tasks),
            desc="generating train set",
        ):
            if res is not None:
                samples.append(res)

        val_tasks = []
        tok = ChessTokenizer()
        for v_idx in range(val_positions_count * 2):
            if len(val_tasks) >= int(val_positions_count * 1.15):
                break
            if samples and v_idx % 2 == 0:
                s_item = samples[(v_idx * 7) % len(samples)]
                seq_ids = s_item["input_ids"].tolist()
                last_tgt = int(s_item["target_ids"][-1])
                full_ids = seq_ids[1:] + ([last_tgt] if last_tgt >= 3 else [])
                moves_uci = [tok.id_to_token[i] for i in full_ids if i >= 3]
            else:
                moves_uci = list(openings[(seed + 50000 + v_idx * 31) % len(openings)])

            if len(moves_uci) < 8:
                continue
            target_ply = rng.randint(8, len(moves_uci))
            val_tasks.append((len(val_tasks), moves_uci, target_ply))

        val_positions = []
        for v_res in tqdm(
            pool.imap_unordered(_worker_build_val_position, val_tasks, chunksize=4),
            total=len(val_tasks),
            desc="annotating val set",
        ):
            if v_res is not None:
                val_positions.append(v_res)
                if len(val_positions) >= val_positions_count:
                    break

    rng.shuffle(samples)
    val_positions = val_positions[:val_positions_count]
    for i, vp in enumerate(val_positions):
        vp["id"] = i

    pillar_counts = {
        "pillar_1_full_games": sum(1 for s in samples if int(s["pillar"]) == 0),
        "pillar_2_tactical_weighted": sum(1 for s in samples if int(s["pillar"]) == 1),
        "pillar_3_legal_regularization": sum(1 for s in samples if int(s["pillar"]) == 2),
    }
    total_plies = sum(len(s["input_ids"]) for s in samples)

    train_path = out_dir / "train_datamix.pt"
    val_path = out_dir / "val_positions.pt"
    torch.save(samples, train_path)
    torch.save(val_positions, val_path)

    summary = {
        "train_path": str(train_path),
        "val_path": str(val_path),
        "num_sequences": len(samples),
        "total_plies": int(total_plies),
        "pillar_counts": pillar_counts,
        "pillar_ratios": {k: round(v / max(len(samples), 1), 4) for k, v in pillar_counts.items()},
        "num_val_positions": len(val_positions),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    print(f"[prepare_data] saved {len(samples)} games ({total_plies:,} plies) and {len(val_positions)} val positions in {summary['elapsed_sec']:.1f}s")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(POST_TRAINING_DIR / "data" / "sft_datamix"))
    ap.add_argument("--total_sequences", type=int, default=2000)
    ap.add_argument("--val_positions", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--sf_depth", type=int, default=12)
    ap.add_argument("--sf_time", type=float, default=0.04)
    ap.add_argument("--teacher_level", type=int, default=2)
    ap.add_argument("--no_hf", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    build_sft_dataset(
        out_dir=Path(args.out_dir),
        total_sequences=args.total_sequences,
        val_positions_count=args.val_positions,
        workers=args.workers,
        sf_depth=args.sf_depth,
        sf_time_limit=args.sf_time,
        teacher_level=args.teacher_level,
        use_hf_dataset=not args.no_hf,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
