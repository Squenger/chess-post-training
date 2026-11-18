"""Fixed move-level tokenizer.

Every token is one chess move in UCI notation (e.g. ``e2e4``, ``e1g1`` for
castling, ``e7e8q`` for a promotion). The vocabulary contains the 1968 moves
that are geometrically possible on a chessboard, so every legal move in every
position has exactly one token. There is no unknown token.

A game is encoded as ``[BOS] m1 m2 m3 ...``. The side to move is implied by
the parity of the position: index 1 is White's first move, index 2 Black's...

DO NOT MODIFY: the evaluation uses this exact file.
"""

from __future__ import annotations

import json
from typing import Iterable, List, Sequence

PAD, BOS, EOS = "[PAD]", "[BOS]", "[EOS]"
SPECIAL_TOKENS = [PAD, BOS, EOS]

_FILES = "abcdefgh"


def _square_name(sq: int) -> str:
    return _FILES[sq % 8] + str(sq // 8 + 1)


def _all_moves() -> List[str]:
    moves = []
    for f in range(64):
        fr, ff = divmod(f, 8)
        for t in range(64):
            if f == t:
                continue
            tr, tf = divmod(t, 8)
            dr, df = abs(tr - fr), abs(tf - ff)
            uci = _square_name(f) + _square_name(t)
            if dr == 0 or df == 0 or dr == df or (dr, df) in ((1, 2), (2, 1)):
                moves.append(uci)
            if df <= 1 and ((fr == 6 and tr == 7) or (fr == 1 and tr == 0)):
                moves.extend(uci + p for p in "qrbn")
    return sorted(moves)


class ChessTokenizer:
    def __init__(self):
        self.id_to_token: List[str] = SPECIAL_TOKENS + _all_moves()
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}
        self.pad_token_id = self.token_to_id[PAD]
        self.bos_token_id = self.token_to_id[BOS]
        self.eos_token_id = self.token_to_id[EOS]
        self.n_special = len(SPECIAL_TOKENS)

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, moves: Iterable[str] | str, add_bos: bool = True) -> List[int]:
        """Encode a list of UCI moves (or a space-separated string)."""
        if isinstance(moves, str):
            moves = moves.split()
        ids = [self.bos_token_id] if add_bos else []
        ids.extend(self.token_to_id[m] for m in moves)
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> List[str]:
        toks = [self.id_to_token[int(i)] for i in ids]
        if skip_special:
            toks = [t for t in toks if t not in SPECIAL_TOKENS]
        return toks

    def is_move_id(self, i: int) -> bool:
        return int(i) >= self.n_special

    def save_vocab(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.token_to_id, f, indent=0)
