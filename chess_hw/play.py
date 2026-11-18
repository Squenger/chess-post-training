"""The fixed inference procedure: how a model chooses a move.

DO NOT MODIFY: this is exactly what the evaluation runs.

Given the move history, the model does ONE forward pass on
``[BOS] m1 ... mk`` and we look at the next-token logits:
  attempt 1: the highest-scoring token,
  attempt 2: the second highest,
  attempt 3: the third highest.
The first candidate that is a legal move is played. If none of the 3 is
legal, a random legal move is played instead (see match.play_game).

The model never sees the list of legal moves; the harness only checks
legality after the fact. Everything is greedy, hence deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import chess
import torch

from .model import ChessGPT
from .tokenizer import ChessTokenizer

MAX_ATTEMPTS = 3
TOKENIZER = ChessTokenizer()


@dataclass
class MoveChoice:
    move: Optional[chess.Move]  # None = all attempts illegal -> random legal move
    attempt: int  # 1, 2 or 3 (0 if all illegal)


class ModelPlayer:
    def __init__(self, model: ChessGPT, name: str = "model"):
        self.model = model.eval()
        self.name = name
        self.device = next(model.parameters()).device

    @torch.inference_mode()
    def candidates(self, history: Sequence[str]) -> List[str]:
        ids = torch.tensor([TOKENIZER.encode(history)], device=self.device)
        logits = self.model(ids)[0, -1].float().cpu()
        order = torch.sort(logits, descending=True, stable=True).indices[:MAX_ATTEMPTS]
        return [TOKENIZER.id_to_token[int(i)] for i in order]

    def choose(self, board: chess.Board) -> MoveChoice:
        history = [m.uci() for m in board.move_stack]
        for attempt, tok in enumerate(self.candidates(history), start=1):
            try:
                # parse_uci checks legality and normalizes castling (e1h1 -> e1g1)
                mv = board.parse_uci(tok)
            except ValueError:  # special token or illegal move
                continue
            return MoveChoice(mv, attempt)
        return MoveChoice(None, 0)
