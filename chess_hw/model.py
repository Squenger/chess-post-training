"""Fixed GPT-style architecture (< 1M parameters).

Plain PyTorch, no `transformers` dependency. The output head is tied to the
token embedding (logits = h @ wte.weight^T), so `wte.weight` is the only
vocabulary-sized tensor.

DO NOT MODIFY: the evaluation builds this exact architecture and only loads
your `model.safetensors` into it. Any change to the architecture (shapes,
names, number of tensors, dtype) makes your submission invalid.
You may of course subclass / wrap it for training (e.g. add a value head in
a separate module), as long as what you save is the state dict of
`ChessGPT` itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ChessGPTConfig:
    vocab_size: int = 1971  # 3 special tokens + 1968 UCI moves
    n_ctx: int = 256
    n_embd: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_inner: int = 384
    layer_norm_epsilon: float = 1e-5


CONFIG = ChessGPTConfig()


class Attention(nn.Module):
    def __init__(self, cfg: ChessGPTConfig, dropout: float):
        super().__init__()
        self.n_head = cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg: ChessGPTConfig, dropout: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.attn = Attention(cfg, dropout)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, cfg.n_inner),
            nn.GELU(),
            nn.Linear(cfg.n_inner, cfg.n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class ChessGPT(nn.Module):
    def __init__(self, cfg: ChessGPTConfig = CONFIG, dropout: float = 0.0):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.n_ctx, cfg.n_embd)
        self.drop = nn.Dropout(dropout)
        self.h = nn.ModuleList([Block(cfg, dropout) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Final hidden states (after ln_f), shape (B, T, n_embd).
        Handy if you want to train an auxiliary head (value, ...) on top."""
        T = input_ids.shape[1]
        assert T <= self.cfg.n_ctx, f"sequence length {T} > n_ctx {self.cfg.n_ctx}"
        pos = torch.arange(T, device=input_ids.device)
        x = self.drop(self.wte(input_ids) + self.wpe(pos))
        for block in self.h:
            x = block(x)
        return self.ln_f(x)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return next-token logits of shape (B, T, vocab_size).

        Attention is causal, so sequences of different lengths can be batched
        with right padding: the logits at position t only depend on tokens <= t.
        """
        return F.linear(self.hidden_states(input_ids), self.wte.weight)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Serialization: a submission is a single `model.safetensors` file holding the
# state dict of ChessGPT, float32, with exactly the reference keys and shapes.
# ---------------------------------------------------------------------------

def reference_shapes() -> Dict[str, tuple]:
    return {k: tuple(v.shape) for k, v in ChessGPT().state_dict().items()}


def validate_state_dict(sd: Dict[str, torch.Tensor]) -> Optional[str]:
    """Return None if `sd` matches the reference architecture, else an error message."""
    ref = reference_shapes()
    missing = sorted(set(ref) - set(sd))
    extra = sorted(set(sd) - set(ref))
    if missing or extra:
        return f"key mismatch: missing={missing[:5]} extra={extra[:5]}"
    for k, shape in ref.items():
        t = sd[k]
        if tuple(t.shape) != shape:
            return f"shape mismatch for {k}: {tuple(t.shape)} != {shape}"
        if t.dtype != torch.float32:
            return f"dtype of {k} is {t.dtype}, expected torch.float32"
        if not torch.isfinite(t).all():
            return f"non-finite values in {k}"
    return None


def save_model(model: ChessGPT, path: str | Path) -> Path:
    """Save the weights to `path` (a directory or a .safetensors file)."""
    from safetensors.torch import save_file

    path = Path(path)
    if path.suffix != ".safetensors":
        path.mkdir(parents=True, exist_ok=True)
        path = path / "model.safetensors"
    sd = {k: v.detach().to("cpu", torch.float32).contiguous() for k, v in model.state_dict().items()}
    err = validate_state_dict(sd)
    if err:
        raise ValueError(err)
    save_file(sd, str(path), metadata={"format": "pt", "arch": "ChessGPT", **{k: str(v) for k, v in asdict(CONFIG).items()}})
    return path


def check_not_lfs_pointer(path: str | Path) -> None:
    """Fail clearly if `path` is a git-lfs pointer (repo cloned without git-lfs)."""
    with open(path, "rb") as f:
        if f.read(24) == b"version https://git-lfs.":
            raise RuntimeError(
                f"{path} is a git-lfs pointer, not the real file: the repository was cloned without git-lfs. "
                "Run `git lfs install && git lfs pull` in the repository (or use `uvx hf download ...`, see README).")


def load_model(path: str | Path, device: str = "cpu") -> ChessGPT:
    """Load weights from a directory containing model.safetensors, or a .safetensors file."""
    from safetensors.torch import load_file

    path = Path(path)
    if path.is_dir():
        path = path / "model.safetensors"
    check_not_lfs_pointer(path)
    sd = load_file(str(path))
    err = validate_state_dict(sd)
    if err:
        raise ValueError(f"{path}: {err}")
    model = ChessGPT()
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()
