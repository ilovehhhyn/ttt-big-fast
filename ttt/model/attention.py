"""Sliding-window causal attention with a rolling KV cache.

Mirrors the `SWA` class of test-time-training/e2e (`ttt/model/attention.py`), but in
PyTorch. Sequences are consumed chunk by chunk because the TTT inner loop updates the
MLP weights once per chunk, so attention must keep the last `window_size` keys/values
around and let each chunk attend to `[cache, chunk]`.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from ttt.config import ModelConfig
from ttt.model.mlp import make_projection
from ttt.model.rope import apply_rope

__all__ = ["KVCache", "SlidingWindowAttention", "build_swa_mask"]


@dataclass
class KVCache:
    """Last `window` keys/values, left-padded.

    The buffer has a fixed length W. Valid entries live in the *last* `length` slots,
    i.e. slots ``[W - length, W)``; slot ``j`` holds the key at position
    ``chunk_start - (W - j)``. There is no ring-buffer wraparound: the buffer is
    rebuilt by ``cat([old, new])[:, -W:]`` every chunk, which keeps the layout
    contiguous and keeps autograd happy across chunks.
    """

    k: Tensor  # [B, W, n_kv, D]
    v: Tensor  # [B, W, n_kv, D]
    length: Tensor  # scalar int64 tensor: min(W, tokens seen so far)

    @staticmethod
    def empty(batch: int, window: int, n_kv: int, head_dim: int, *, device, dtype) -> KVCache:
        assert batch > 0 and window > 0 and n_kv > 0 and head_dim > 0
        shape = (batch, window, n_kv, head_dim)
        return KVCache(
            k=torch.zeros(shape, device=device, dtype=dtype),
            v=torch.zeros(shape, device=device, dtype=dtype),
            length=torch.zeros((), device=device, dtype=torch.int64),
        )


def build_swa_mask(
    q_pos: Tensor,
    k_pos: Tensor,
    window_size: int,
    k_valid: Tensor | None = None,
) -> Tensor:
    """Sliding-window causal mask, ``True`` = "this query may attend this key".

        allowed(q, k)  iff  k <= q  and  q - k < window_size   (and key slot is valid)

    q_pos: [Tq] integer positions of the queries (absolute, or relative to any common origin).
    k_pos: [Tk] integer positions of the keys, on the same axis as q_pos.
    k_valid: [Tk] bool, False for cache slots that have never been written.
    Returns [Tq, Tk] bool.
    """
    assert q_pos.ndim == 1 and k_pos.ndim == 1, "positions must be 1-D"
    assert window_size >= 1, f"window_size must be >= 1, got {window_size}"
    delta = q_pos[:, None] - k_pos[None, :]
    mask = (delta >= 0) & (delta < window_size)
    if k_valid is not None:
        assert k_valid.shape == k_pos.shape, "k_valid must match k_pos"
        mask = mask & k_valid[None, :]
    return mask


class SlidingWindowAttention(nn.Module):
    """Grouped-query sliding-window attention.

        attn(Q, K, V) = softmax( Q K^T / sqrt(head_dim) + mask ) V

    with `num_kv_heads` key/value heads each shared by `num_heads // num_kv_heads`
    query heads (GQA), and the sliding-window causal mask of `build_swa_mask`.

    `use_math_backend=True` forces `SDPBackend.MATH`, the only SDPA backend that is
    double-backward capable. Blocks at or above `cfg.first_fast_layer` need it because
    the TTT outer loop differentiates through the inner-loop gradient.
    """

    def __init__(self, cfg: ModelConfig, *, use_math_backend: bool) -> None:
        super().__init__()
        assert cfg.num_heads % cfg.num_kv_heads == 0, (
            f"GQA needs num_heads ({cfg.num_heads}) divisible by num_kv_heads ({cfg.num_kv_heads})"
        )
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.n_rep = cfg.num_heads // cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.hidden_size = cfg.hidden_size
        self.window_size = cfg.window_size
        self.use_math_backend = use_math_backend

        q_dim = cfg.num_heads * cfg.head_dim
        kv_dim = cfg.num_kv_heads * cfg.head_dim
        self.wq = make_projection("wq", cfg.hidden_size, q_dim, cfg.lora)
        self.wk = make_projection("wk", cfg.hidden_size, kv_dim, cfg.lora)
        self.wv = make_projection("wv", cfg.hidden_size, kv_dim, cfg.lora)
        self.wo = make_projection("wo", q_dim, cfg.hidden_size, cfg.lora)

    # ------------------------------------------------------------------ helpers

    def _sdpa_context(self):
        # MATH is the only SDPA backend with a double-backward formula.
        return sdpa_kernel([SDPBackend.MATH]) if self.use_math_backend else contextlib.nullcontext()

    def _repeat_kv(self, x: Tensor) -> Tensor:
        """[B, T, n_kv, D] -> [B, T, n_heads, D]; kv head g serves query heads [g*n_rep, (g+1)*n_rep)."""
        if self.n_rep == 1:
            return x
        return torch.repeat_interleave(x, self.n_rep, dim=2)

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        cache: KVCache | None,
    ) -> tuple[Tensor, KVCache | None]:
        """x: [B, T, hidden]; cos/sin: [T, head_dim//2] for this chunk's absolute positions.

        cache is None  -> plain causal sliding-window attention over the whole T
                          (the frozen prefix sees the full sequence at once).
        cache is given -> queries of this chunk attend cat([cache, chunk]); the updated
                          cache (last `window_size` positions) is returned.
        """
        assert x.ndim == 3, f"expected [B, T, hidden], got {tuple(x.shape)}"
        b, t, hidden = x.shape
        assert hidden == self.hidden_size, f"expected hidden {self.hidden_size}, got {hidden}"
        d, w = self.head_dim, self.window_size

        q = self.wq(x).view(b, t, self.num_heads, d)
        k = self.wk(x).view(b, t, self.num_kv_heads, d)
        v = self.wv(x).view(b, t, self.num_kv_heads, d)

        # RoPE is applied before caching, so cached keys already carry their absolute phase.
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is None:
            positions = torch.arange(t, device=x.device)
            mask = build_swa_mask(positions, positions, w)
            k_all, v_all = k, v
            new_cache = None
        else:
            assert t <= w, f"chunk length {t} exceeds window_size {w}"
            assert cache.k.shape == (b, w, self.num_kv_heads, d), (
                f"cache.k must be {(b, w, self.num_kv_heads, d)}, got {tuple(cache.k.shape)}"
            )
            assert cache.v.shape == cache.k.shape, "cache.k and cache.v must match"
            k_all = torch.cat([cache.k.to(k.dtype), k], dim=1)  # [B, W + T, n_kv, D]
            v_all = torch.cat([cache.v.to(v.dtype), v], dim=1)

            # Positions relative to this chunk's start: cache slot j -> j - W, chunk i -> i.
            q_pos = torch.arange(t, device=x.device)
            k_pos = torch.arange(-w, t, device=x.device)
            slot = torch.arange(w, device=x.device)
            k_valid = torch.cat(
                [slot >= (w - cache.length.to(slot.device)), torch.ones(t, dtype=torch.bool, device=x.device)]
            )
            mask = build_swa_mask(q_pos, k_pos, w, k_valid)

            # Roll the window forward: keep the last W positions of [cache, chunk].
            new_cache = KVCache(
                k=k_all[:, -w:],
                v=v_all[:, -w:],
                length=torch.clamp(cache.length + t, max=w),
            )

        # GQA: expand each kv head to n_rep query heads, then [B, H, T, D] for SDPA.
        k_all = self._repeat_kv(k_all)
        v_all = self._repeat_kv(v_all)
        q = q.transpose(1, 2)
        k_all = k_all.transpose(1, 2)
        v_all = v_all.transpose(1, 2)

        # softmax(QK^T / sqrt(D) + mask) V, computed in the input dtype.
        with self._sdpa_context():
            out = F.scaled_dot_product_attention(
                q, k_all, v_all, attn_mask=mask[None, None], scale=1.0 / math.sqrt(d)
            )

        out = out.transpose(1, 2).reshape(b, t, self.num_heads * d)
        return self.wo(out), new_cache
