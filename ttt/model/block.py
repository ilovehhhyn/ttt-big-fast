"""Llama-style pre-norm transformer block."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from ttt.config import ModelConfig
from ttt.model.attention import KVCache, SlidingWindowAttention
from ttt.model.mlp import SwiGLUMLP

__all__ = ["TransformerBlock"]


class TransformerBlock(nn.Module):
    """Pre-norm residual block, exactly as in Llama-3.2:

        x = x + attn(seq_norm(x))
        x = x + mlp(ffn_norm(x))

    No post-norm and no QK-norm: Llama-3.2 has neither.

    `use_math_backend` is forwarded to the attention module; blocks at or above
    `cfg.first_fast_layer` must set it so the TTT outer loop can backprop through
    the inner-loop gradient (double backward).
    """

    def __init__(self, cfg: ModelConfig, *, use_math_backend: bool) -> None:
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.seq_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.attn = SlidingWindowAttention(cfg, use_math_backend=use_math_backend)
        self.ffn_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = SwiGLUMLP(cfg.hidden_size, cfg.intermediate_size, cfg.lora)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        cache: KVCache | None,
    ) -> tuple[Tensor, KVCache | None]:
        assert x.ndim == 3, f"expected [B, T, hidden], got {tuple(x.shape)}"
        assert x.shape[-1] == self.hidden_size, f"expected hidden {self.hidden_size}, got {x.shape[-1]}"
        attn_out, new_cache = self.attn(self.seq_norm(x), cos, sin, cache)
        h = x + attn_out
        return h + self.mlp(self.ffn_norm(h)), new_cache
