"""Rotary position embeddings with optional Llama-3 frequency rescaling.

Conventions match HuggingFace's `LlamaRotaryEmbedding` / `apply_rotary_pos_emb`
exactly (verified in tests/test_rope.py): the *halves* convention, not the
interleaved-adjacent-pairs convention of the original RoFormer paper.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ttt.config import RopeConfig

__all__ = ["build_rope_cache", "apply_rope"]


def build_rope_cache(
    head_dim: int,
    max_seq: int,
    cfg: RopeConfig,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Return (cos, sin), each ``[max_seq, head_dim // 2]``.

    Base frequencies:                f_j = 1 / theta^(2j / head_dim),  j = 0 .. D/2-1
    Angle for position p:            phi_{p,j} = p * f_j
                                     cos[p, j] = cos(phi), sin[p, j] = sin(phi)

    Llama-3 rescaling rule, in terms of the wavelength lambda_j = 2*pi / f_j and
    the pretraining context length L = cfg.original_max_position:

        lambda_j > L / low_freq_factor    (low  frequency)  ->  f_j / factor
        lambda_j < L / high_freq_factor   (high frequency)  ->  f_j
        otherwise                         (medium)          ->  smooth blend
            s = (L / lambda_j - low_freq_factor) / (high_freq_factor - low_freq_factor)
            f_j <- (1 - s) * (f_j / factor) + s * f_j

    Always computed in float32 for precision, then cast to `dtype`.
    """
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
    assert max_seq > 0, f"max_seq must be positive, got {max_seq}"

    half = head_dim // 2
    exponent = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
    inv_freq = 1.0 / (cfg.theta**exponent)  # [D/2], float32

    if cfg.scaling == "llama3":
        inv_freq = _llama3_rescale(inv_freq, cfg)
    else:
        assert cfg.scaling == "none", f"unknown rope scaling {cfg.scaling!r}"

    positions = torch.arange(max_seq, device=device, dtype=torch.float32)  # [T]
    angles = torch.outer(positions, inv_freq)  # [T, D/2]
    assert angles.shape == (max_seq, half)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def _llama3_rescale(inv_freq: Tensor, cfg: RopeConfig) -> Tensor:
    """Apply the Llama-3 wavelength-dependent frequency rescaling (see docstring above)."""
    orig = float(cfg.original_max_position)
    low_wavelen = orig / cfg.low_freq_factor
    high_wavelen = orig / cfg.high_freq_factor
    wavelen = 2.0 * math.pi / inv_freq

    scaled = torch.where(wavelen > low_wavelen, inv_freq / cfg.factor, inv_freq)
    smooth = (orig / wavelen - cfg.low_freq_factor) / (cfg.high_freq_factor - cfg.low_freq_factor)
    blended = (1.0 - smooth) * scaled / cfg.factor + smooth * scaled
    is_medium = ~(wavelen < high_wavelen) & ~(wavelen > low_wavelen)
    return torch.where(is_medium, blended, scaled)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor, *, interleaved: bool = False) -> Tensor:
    """Rotate `x` by the angles in (cos, sin).

    x:       [B, T, H, D]
    cos/sin: [T, D//2], already gathered for this chunk's absolute positions.

    Two conventions, differing in WHICH pairs of channels form each 2-D rotation.
    They are not interchangeable: a model trained under one scores badly under the other.

    interleaved=False - halves (Llama / HuggingFace `rotate_half`):
        x1 = x[..., :D//2], x2 = x[..., D//2:]
        out = cat(x1 * cos - x2 * sin,  x2 * cos + x1 * sin)

    interleaved=True - adjacent pairs (TTT-E2E, and the classic GPT-J form). Their
    `apply_rotary_emb` does `x.reshape(..., -1, 2)` and treats the two entries as the
    real and imaginary parts of a complex number, i.e. channels (0,1), (2,3), ... pair up:
        out[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
        out[..., 1::2] = x[..., 1::2] * cos + x[..., 0::2] * sin
    """
    assert x.ndim == 4, f"apply_rope expects [B, T, H, D], got {tuple(x.shape)}"
    b, t, h, d = x.shape
    assert d % 2 == 0, f"head dim must be even, got {d}"
    half = d // 2
    assert cos.shape == (t, half), f"cos must be [T, D//2] = {(t, half)}, got {tuple(cos.shape)}"
    assert sin.shape == (t, half), f"sin must be [T, D//2] = {(t, half)}, got {tuple(sin.shape)}"

    cos_b = cos.to(x.dtype).view(1, t, 1, half)
    sin_b = sin.to(x.dtype).view(1, t, 1, half)
    if interleaved:
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        out = torch.stack([x_even * cos_b - x_odd * sin_b,
                           x_odd * cos_b + x_even * sin_b], dim=-1).flatten(-2)
    else:
        x1, x2 = x[..., :half], x[..., half:]
        out = torch.cat([x1 * cos_b - x2 * sin_b, x2 * cos_b + x1 * sin_b], dim=-1)
    assert out.shape == (b, t, h, d)
    return out
