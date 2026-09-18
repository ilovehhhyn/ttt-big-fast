"""Llama-style pre-norm transformer block."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from ttt.config import ModelConfig
from ttt.model.attention import KVCache, SlidingWindowAttention
from ttt.model.mlp import SwiGLUMLP

__all__ = ["TransformerBlock"]


class TransformerBlock(nn.Module):
    """Residual block. Llama-3.2 (default) is plain pre-norm:

        x = x + attn(seq_norm(x))
        x = x + mlp(ffn_norm(x))

    With cfg.post_norm (TTT-E2E) each sublayer output is normalised as well:

        x = x + seq_post_norm(attn(seq_norm(x)))
        x = x + ffn_post_norm(mlp(ffn_norm(x)))

    With cfg.prime (TTT-E2E) a SECOND "prime" MLP is inserted between them and is
    the weight the inner loop updates, leaving the original MLP static as safe
    storage for pretrained knowledge (paper 2.3.1):

        x = x + [post](mlp_prime(ffn_prime_norm(x)))     # inserted here
        x = x + [post](mlp(ffn_norm(x)))

    The prime MLP exists only in suffix (TTT) blocks, so it is created only when
    `is_fast_block` is set.

    `use_math_backend` is forwarded to the attention module; blocks at or above
    `cfg.first_fast_layer` must set it so the TTT outer loop can backprop through
    the inner-loop gradient (double backward).
    """

    def __init__(self, cfg: ModelConfig, *, use_math_backend: bool, is_fast_block: bool = False) -> None:
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.post_norm = cfg.post_norm
        self.seq_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.attn = SlidingWindowAttention(cfg, use_math_backend=use_math_backend)
        self.ffn_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = SwiGLUMLP(cfg.hidden_size, cfg.intermediate_size, cfg.lora)
        if cfg.post_norm:
            self.seq_post_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            self.ffn_post_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.has_prime = bool(cfg.prime and is_fast_block)
        if self.has_prime:
            self.ffn_prime_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            self.mlp_prime = SwiGLUMLP(cfg.hidden_size, cfg.intermediate_size, cfg.lora)
            if cfg.post_norm:
                self.ffn_prime_post_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

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
        if self.post_norm:
            attn_out = self.seq_post_norm(attn_out)
        h = x + attn_out

        if self.has_prime:
            prime_out = self.mlp_prime(self.ffn_prime_norm(h))
            if self.post_norm:
                prime_out = self.ffn_prime_post_norm(prime_out)
            h = h + prime_out

        ffn_out = self.mlp(self.ffn_norm(h))
        if self.post_norm:
            ffn_out = self.ffn_post_norm(ffn_out)
        return h + ffn_out, new_cache
