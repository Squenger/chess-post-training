"""Deterministic opponent engine (pure python-chess, no Stockfish).

Levels:
  0  uniformly random legal move (seeded, so still reproducible)
  1  1-ply alpha-beta + quiescence search
  2  2-ply alpha-beta + quiescence search
  3  3-ply alpha-beta + quiescence search
  4  4-ply alpha-beta + quiescence search (slow: ~seconds per move)

Evaluation: material + piece-square tables (simplified PeSTO middlegame
values). Move ordering and tie-breaking are fully deterministic, so for a
given position and level the engine always plays the same move.
"""

from __future__ import annotations

import random
from typing import Optional

import chess

MATE = 100_000
VAL = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}

# Tables are written from White's point of view, a8..h8 first (rank 8 on top).
_PST_TXT = {
    chess.PAWN: """
      0   0   0   0   0   0   0   0
     50  50  50  50  50  50  50  50
     10  10  20  30  30  20  10  10
      5   5  10  25  25  10   5   5
      0   0   0  20  20   0   0   0
      5  -5 -10   0   0 -10  -5   5
      5  10  10 -20 -20  10  10   5
      0   0   0   0   0   0   0   0""",
    chess.KNIGHT: """
    -50 -40 -30 -30 -30 -30 -40 -50
    -40 -20   0   0   0   0 -20 -40
    -30   0  10  15  15  10   0 -30
    -30   5  15  20  20  15   5 -30
    -30   0  15  20  20  15   0 -30
    -30   5  10  15  15  10   5 -30
    -40 -20   0   5   5   0 -20 -40
    -50 -40 -30 -30 -30 -30 -40 -50""",
    chess.BISHOP: """
    -20 -10 -10 -10 -10 -10 -10 -20
    -10   0   0   0   0   0   0 -10
    -10   0   5  10  10   5   0 -10
    -10   5   5  10  10   5   5 -10
    -10   0  10  10  10  10   0 -10
    -10  10  10  10  10  10  10 -10
    -10   5   0   0   0   0   5 -10
    -20 -10 -10 -10 -10 -10 -10 -20""",
    chess.ROOK: """
      0   0   0   0   0   0   0   0
      5  10  10  10  10  10  10   5
     -5   0   0   0   0   0   0  -5
     -5   0   0   0   0   0   0  -5
     -5   0   0   0   0   0   0  -5
     -5   0   0   0   0   0   0  -5
     -5   0   0   0   0   0   0  -5
      0   0   0   5   5   0   0   0""",
    chess.QUEEN: """
    -20 -10 -10  -5  -5 -10 -10 -20
    -10   0   0   0   0   0   0 -10
    -10   0   5   5   5   5   0 -10
     -5   0   5   5   5   5   0  -5
      0   0   5   5   5   5   0  -5
    -10   5   5   5   5   5   0 -10
    -10   0   5   0   0   0   0 -10
    -20 -10 -10  -5  -5 -10 -10 -20""",
    chess.KING: """
    -30 -40 -40 -50 -50 -40 -40 -30
    -30 -40 -40 -50 -50 -40 -40 -30
    -30 -40 -40 -50 -50 -40 -40 -30
    -30 -40 -40 -50 -50 -40 -40 -30
    -20 -30 -30 -40 -40 -30 -30 -20
    -10 -20 -20 -20 -20 -20 -20 -10
     20  20   0   0   0   0  20  20
     20  30  10   0   0  10  30  20""",
}


def _parse(txt: str):
    rows = [list(map(int, r.split())) for r in txt.strip().splitlines()]
    tbl = [0] * 64
    for r, row in enumerate(rows):  # r=0 is rank 8
        for f, v in enumerate(row):
            tbl[(7 - r) * 8 + f] = v
    return tbl


# PST[color][piece_type][square] including material
PST = {chess.WHITE: {}, chess.BLACK: {}}
for _pt, _txt in _PST_TXT.items():
    _w = _parse(_txt)
    PST[chess.WHITE][_pt] = [VAL[_pt] + v for v in _w]
    PST[chess.BLACK][_pt] = [VAL[_pt] + _w[chess.square_mirror(s)] for s in range(64)]


def evaluate(board: chess.Board) -> int:
    """Static evaluation from the side-to-move's point of view."""
    score = 0
    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        tables = PST[color]
        for pt in range(1, 7):
            tbl = tables[pt]
            bb = board.pieces_mask(pt, color)
            while bb:
                sq = (bb & -bb).bit_length() - 1
                score += sign * tbl[sq]
                bb &= bb - 1
    return score if board.turn == chess.WHITE else -score


def _victim_value(board: chess.Board, m: chess.Move) -> int:
    if board.is_en_passant(m):
        return VAL[chess.PAWN]
    p = board.piece_type_at(m.to_square)
    return VAL[p] if p else 0


def _ordered(board: chess.Board, captures_only: bool = False):
    caps, quiet = [], []
    gen = board.generate_legal_captures() if captures_only else board.legal_moves
    for m in gen:
        if board.is_capture(m) or m.promotion:
            score = 10 * _victim_value(board, m) - VAL[board.piece_type_at(m.from_square)] // 10
            if m.promotion:
                score += VAL[m.promotion]
            caps.append((-score, m.uci(), m))
        elif not captures_only:
            quiet.append((m.uci(), m))
    caps.sort(key=lambda t: (t[0], t[1]))
    quiet.sort(key=lambda t: t[0])
    return [t[2] for t in caps] + [t[1] for t in quiet]


class Engine:
    def __init__(self, level: int = 2, seed: int = 0, qdepth: int = 4):
        if not 0 <= level <= 4:
            raise ValueError("level must be in 0..4")
        self.level = level
        self.qdepth = qdepth
        self.rng = random.Random(seed)
        self.nodes = 0

    def _quiesce(self, board, alpha, beta, qd):
        self.nodes += 1
        stand = evaluate(board)
        if stand >= beta:
            return beta
        if alpha < stand:
            alpha = stand
        if qd == 0:
            return alpha
        for m in _ordered(board, captures_only=True):
            board.push(m)
            s = -self._quiesce(board, -beta, -alpha, qd - 1)
            board.pop()
            if s >= beta:
                return beta
            if s > alpha:
                alpha = s
        return alpha

    def _search(self, board, depth, alpha, beta, ply):
        self.nodes += 1
        moves = _ordered(board)
        if not moves:
            return -MATE + ply if board.is_check() else 0
        if board.halfmove_clock >= 100 or board.is_insufficient_material():
            return 0
        if depth == 0:
            return self._quiesce(board, alpha, beta, self.qdepth)
        for m in moves:
            board.push(m)
            s = -self._search(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if s >= beta:
                return beta
            if s > alpha:
                alpha = s
        return alpha

    def best_move(self, board: chess.Board) -> Optional[chess.Move]:
        moves = _ordered(board)
        if not moves:
            return None
        if self.level == 0:
            return self.rng.choice(sorted(moves, key=lambda m: m.uci()))
        board = board.copy(stack=False)
        best, best_score = moves[0], -MATE - 1
        alpha, beta = -MATE - 1, MATE + 1
        for m in moves:
            board.push(m)
            s = -self._search(board, self.level - 1, -beta, -alpha, 1)
            board.pop()
            if s > best_score:
                best, best_score = m, s
            if s > alpha:
                alpha = s
        return best
