"""RoPE unit tests, checked against HuggingFace's reference Llama implementation."""

import pytest
import torch

from ttt.config import RopeConfig
from ttt.model.rope import apply_rope, build_rope_cache

HEAD_DIM = 64
SEQ = 128
THETA = 500000.0
TOL = 1e-5

LLAMA3_KWARGS = {
    "rope_type": "llama3",
    "factor": 32.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}


def _hf_rope(scaling: str):
    """Build HF's LlamaRotaryEmbedding for the same settings we use."""
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    kwargs = dict(
        hidden_size=HEAD_DIM * 4,
        num_attention_heads=4,
        head_dim=HEAD_DIM,
        max_position_embeddings=131072,
        rope_theta=THETA,
    )
    if scaling == "llama3":
        kwargs["rope_scaling"] = dict(LLAMA3_KWARGS)
    return LlamaRotaryEmbedding(LlamaConfig(**kwargs))


@pytest.mark.parametrize("scaling", ["none", "llama3"])
def test_rope_matches_hf(scaling):
    """build_rope_cache + apply_rope == HF LlamaRotaryEmbedding + apply_rotary_pos_emb."""
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    cfg = RopeConfig(
        theta=THETA,
        scaling=scaling,
        factor=LLAMA3_KWARGS["factor"],
        low_freq_factor=LLAMA3_KWARGS["low_freq_factor"],
        high_freq_factor=LLAMA3_KWARGS["high_freq_factor"],
        original_max_position=LLAMA3_KWARGS["original_max_position_embeddings"],
    )
    cos, sin = build_rope_cache(HEAD_DIM, SEQ, cfg)
    assert cos.shape == (SEQ, HEAD_DIM // 2)
    assert sin.shape == (SEQ, HEAD_DIM // 2)
    assert cos.dtype is torch.float32

    rot = _hf_rope(scaling)
    b, n_heads = 2, 4
    x = torch.randn(b, SEQ, n_heads, HEAD_DIM)
    position_ids = torch.arange(SEQ).unsqueeze(0).expand(b, -1)
    hf_cos, hf_sin = rot(x, position_ids)  # [B, T, head_dim] (halves duplicated)

    # HF duplicates the D//2 frequencies: cat(freqs, freqs).
    torch.testing.assert_close(torch.cat([cos, cos], dim=-1), hf_cos[0], atol=TOL, rtol=TOL)
    torch.testing.assert_close(torch.cat([sin, sin], dim=-1), hf_sin[0], atol=TOL, rtol=TOL)

    q = torch.randn(b, SEQ, n_heads, HEAD_DIM)
    k = torch.randn(b, SEQ, 2, HEAD_DIM)
    # unsqueeze_dim=2 because our layout is [B, T, H, D].
    hf_q, hf_k = apply_rotary_pos_emb(q, k, hf_cos, hf_sin, unsqueeze_dim=2)

    torch.testing.assert_close(apply_rope(q, cos, sin), hf_q, atol=TOL, rtol=TOL)
    torch.testing.assert_close(apply_rope(k, cos, sin), hf_k, atol=TOL, rtol=TOL)


def test_rope_position_slice_is_translation_invariant():
    """Slicing the cache for absolute positions must equal building it for those positions."""
    cfg = RopeConfig(theta=THETA, scaling="none")
    cos, sin = build_rope_cache(HEAD_DIM, SEQ, cfg)
    x = torch.randn(1, 8, 2, HEAD_DIM)
    start = 40
    full = apply_rope(x, cos[start : start + 8], sin[start : start + 8])
    # Same rotation, computed one position at a time.
    per_pos = torch.cat(
        [apply_rope(x[:, i : i + 1], cos[start + i : start + i + 1], sin[start + i : start + i + 1]) for i in range(8)],
        dim=1,
    )
    torch.testing.assert_close(full, per_pos)


def test_rope_preserves_dot_products_of_equal_positions():
    """RoPE is a rotation: it preserves norms."""
    cfg = RopeConfig(theta=THETA, scaling="llama3")
    cos, sin = build_rope_cache(HEAD_DIM, SEQ, cfg)
    x = torch.randn(3, SEQ, 4, HEAD_DIM, dtype=torch.float64)
    y = apply_rope(x, cos.double(), sin.double())
    torch.testing.assert_close(x.norm(dim=-1), y.norm(dim=-1))


def test_rope_cache_dtype_cast():
    cfg = RopeConfig(theta=THETA, scaling="none")
    cos, sin = build_rope_cache(HEAD_DIM, 16, cfg, dtype=torch.bfloat16)
    assert cos.dtype is torch.bfloat16 and sin.dtype is torch.bfloat16
