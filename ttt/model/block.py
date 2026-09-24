"""Llama-style pre-norm transformer block."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ttt.config import ModelConfig
from ttt.model.attention import KVCache, SlidingWindowAttention
from ttt.model.mlp import SwiGLUMLP
from ttt.model.token_rate import TokenRate, scale_gradient

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
    `is_fast_block` is set. It carries no LoRA: arm F trains its weights directly. With
    cfg.prime_gate (arm F) its output is normalised and gated by a scalar that starts at 0:

        x = x + prime_gate * ffn_prime_out_norm(mlp_prime(ffn_prime_norm(x)))

    With cfg.token_rates (LaCT Eq. 4) a fast block also owns a `token_rate` that predicts a
    weight per token from the fast module's normalised input; the weight scales that
    token's contribution to the inner gradient of the fast module and leaves the value
    unchanged (`scale_gradient`).

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
        self.has_prime_gate = bool(self.has_prime and cfg.prime_gate)
        if self.has_prime:
            self.ffn_prime_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            self.mlp_prime = SwiGLUMLP(cfg.hidden_size, cfg.prime_intermediate, None)
            if cfg.post_norm:
                self.ffn_prime_post_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            if cfg.prime_gate:
                self.ffn_prime_out_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
                self.prime_gate = nn.Parameter(torch.zeros(()))
        self.has_token_rate = bool(cfg.token_rates and is_fast_block)
        if self.has_token_rate:
            self.token_rate = TokenRate(hidden_size=cfg.hidden_size)

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
            prime_in = self.ffn_prime_norm(h)
            prime_out = self._rate_write(self.mlp_prime(prime_in), prime_in)
            if self.post_norm:
                prime_out = self.ffn_prime_post_norm(prime_out)
            if self.has_prime_gate:
                prime_out = self.prime_gate * self.ffn_prime_out_norm(prime_out)
            h = h + prime_out

        ffn_in = self.ffn_norm(h)
        ffn_out = self.mlp(ffn_in)
        if not self.has_prime:
            ffn_out = self._rate_write(ffn_out, ffn_in)
        if self.post_norm:
            ffn_out = self.ffn_post_norm(ffn_out)
        return h + ffn_out, new_cache

    def _rate_write(self, fast_out: Tensor, fast_in: Tensor) -> Tensor:
        """Weight each token's write into the fast module by its predicted rate."""
        if not self.has_token_rate:
            return fast_out
        return scale_gradient(fast_out, self.token_rate(fast_in))
