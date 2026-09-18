"""Sliding-window attention: mask correctness, KV-cache chunking, GQA."""

import math
from dataclasses import replace

import pytest
import torch

from ttt.config import LoRAConfig, ModelConfig, RopeConfig
from ttt.model.attention import KVCache, SlidingWindowAttention, build_swa_mask
from ttt.model.rope import apply_rope, build_rope_cache


def _no_lora(cfg: ModelConfig) -> ModelConfig:
    return replace(cfg, lora=LoRAConfig(rank=0))


# --------------------------------------------------------------------------- mask


@pytest.mark.parametrize("window", [1, 2, 3, 5, 8])
def test_sliding_window_mask_exact(window):
    """q attends k iff k <= q and q - k < window. Compared to a hand-written double loop."""
    t = 9
    q_pos = torch.arange(t)
    k_pos = torch.arange(t)
    got = build_swa_mask(q_pos, k_pos, window)
    assert got.shape == (t, t)
    assert got.dtype is torch.bool
    want = torch.zeros(t, t, dtype=torch.bool)
    for i in range(t):
        for j in range(t):
            want[i, j] = (j <= i) and (i - j < window)
    assert torch.equal(got, want), f"\ngot:\n{got.int()}\nwant:\n{want.int()}"


def test_sliding_window_mask_with_cache_positions_and_validity():
    """Cache slots carry absolute positions; invalid (never-written) slots are masked out."""
    window, chunk, start, valid = 6, 3, 10, 4
    # Left-padded cache buffer: slots [window - valid, window) are valid and hold
    # absolute positions start-valid .. start-1.
    k_pos = torch.cat([torch.arange(start - window, start), torch.arange(start, start + chunk)])
    k_valid = torch.cat(
        [torch.arange(window) >= window - valid, torch.ones(chunk, dtype=torch.bool)]
    )
    q_pos = torch.arange(start, start + chunk)

    got = build_swa_mask(q_pos, k_pos, window, k_valid)
    want = torch.zeros(chunk, window + chunk, dtype=torch.bool)
    for i in range(chunk):
        for j in range(window + chunk):
            qp = int(q_pos[i])
            kp = int(k_pos[j])
            want[i, j] = bool(k_valid[j]) and kp <= qp and qp - kp < window
    assert torch.equal(got, want)
    # Every query must see at least itself.
    assert got.any(dim=-1).all()


# --------------------------------------------------------------------------- cache


def test_kvcache_empty_shapes():
    c = KVCache.empty(2, 8, 3, 4, device=torch.device("cpu"), dtype=torch.float32)
    assert c.k.shape == (2, 8, 3, 4)
    assert c.v.shape == (2, 8, 3, 4)
    assert c.length.dtype is torch.int64
    assert int(c.length) == 0


def test_chunked_equals_full(tiny_model_cfg):
    """THE cache test: one-shot over 16 tokens == 4 chunks of 4 with the cache threaded."""
    cfg = _no_lora(tiny_model_cfg)
    assert cfg.window_size == 8 and cfg.chunk_size == 4
    torch.manual_seed(0)
    attn = SlidingWindowAttention(cfg, use_math_backend=True).double()
    seq = 16
    b = 2
    x = torch.randn(b, seq, cfg.hidden_size, dtype=torch.float64)
    cos, sin = build_rope_cache(cfg.head_dim, seq, cfg.rope, dtype=torch.float64)

    full, cache_out = attn(x, cos, sin, None)
    assert cache_out is None
    assert full.shape == (b, seq, cfg.hidden_size)

    cache = KVCache.empty(
        b, cfg.window_size, cfg.num_kv_heads, cfg.head_dim, device=x.device, dtype=x.dtype
    )
    outs = []
    for start in range(0, seq, cfg.chunk_size):
        stop = start + cfg.chunk_size
        y, cache = attn(x[:, start:stop], cos[start:stop], sin[start:stop], cache)
        assert cache is not None
        assert int(cache.length) == min(cfg.window_size, stop)
        outs.append(y)
    chunked = torch.cat(outs, dim=1)

    torch.testing.assert_close(chunked, full, atol=1e-5, rtol=1e-5)


def test_chunked_equals_full_single_chunk_is_identity(tiny_model_cfg):
    """A single chunk fed through an empty cache equals the cache-free path."""
    cfg = _no_lora(tiny_model_cfg)
    attn = SlidingWindowAttention(cfg, use_math_backend=True).double()
    x = torch.randn(1, cfg.chunk_size, cfg.hidden_size, dtype=torch.float64)
    cos, sin = build_rope_cache(cfg.head_dim, cfg.chunk_size, cfg.rope, dtype=torch.float64)
    full, _ = attn(x, cos, sin, None)
    cache = KVCache.empty(1, cfg.window_size, cfg.num_kv_heads, cfg.head_dim, device=x.device, dtype=x.dtype)
    chunked, cache = attn(x, cos, sin, cache)
    torch.testing.assert_close(chunked, full, atol=1e-5, rtol=1e-5)
    assert int(cache.length) == cfg.chunk_size


# --------------------------------------------------------------------------- GQA


def test_gqa_repeat():
    """num_heads=4, num_kv_heads=2 equals manual kv repetition + explicit softmax attention."""
    cfg = ModelConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        window_size=6,
        chunk_size=3,
        fast_blocks=0,
        rope=RopeConfig(theta=10000.0, scaling="none"),
        lora=LoRAConfig(rank=0),
    )
    torch.manual_seed(0)
    attn = SlidingWindowAttention(cfg, use_math_backend=True).double()
    b, t = 2, 7
    x = torch.randn(b, t, cfg.hidden_size, dtype=torch.float64)
    cos, sin = build_rope_cache(cfg.head_dim, t, cfg.rope, dtype=torch.float64)
    got, _ = attn(x, cos, sin, None)

    d = cfg.head_dim
    n_rep = cfg.num_heads // cfg.num_kv_heads
    q = apply_rope(attn.wq(x).view(b, t, cfg.num_heads, d), cos, sin)
    k = apply_rope(attn.wk(x).view(b, t, cfg.num_kv_heads, d), cos, sin)
    v = attn.wv(x).view(b, t, cfg.num_kv_heads, d)
    k = k.repeat_interleave(n_rep, dim=2)
    v = v.repeat_interleave(n_rep, dim=2)
    q, k, v = (z.transpose(1, 2) for z in (q, k, v))  # [B, H, T, D]

    scores = q @ k.transpose(-1, -2) / math.sqrt(d)
    mask = build_swa_mask(torch.arange(t), torch.arange(t), cfg.window_size)
    scores = scores.masked_fill(~mask, float("-inf"))
    out = torch.softmax(scores, dim=-1) @ v
    want = attn.wo(out.transpose(1, 2).reshape(b, t, cfg.num_heads * d))

    torch.testing.assert_close(got, want, atol=1e-10, rtol=1e-10)


def test_projection_shapes(tiny_model_cfg):
    cfg = _no_lora(tiny_model_cfg)
    attn = SlidingWindowAttention(cfg, use_math_backend=False)
    assert attn.wq.weight.shape == (cfg.num_heads * cfg.head_dim, cfg.hidden_size)
    assert attn.wk.weight.shape == (cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size)
    assert attn.wv.weight.shape == (cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size)
    assert attn.wo.weight.shape == (cfg.hidden_size, cfg.num_heads * cfg.head_dim)
    assert attn.wq.bias is None


def test_default_backend_path_runs(tiny_model_cfg):
    """use_math_backend=False must still produce the same numbers (different kernel only)."""
    cfg = _no_lora(tiny_model_cfg)
    torch.manual_seed(0)
    a = SlidingWindowAttention(cfg, use_math_backend=False)
    torch.manual_seed(0)
    b_ = SlidingWindowAttention(cfg, use_math_backend=True)
    x = torch.randn(1, 12, cfg.hidden_size)
    cos, sin = build_rope_cache(cfg.head_dim, 12, cfg.rope)
    ya, _ = a(x, cos, sin, None)
    yb, _ = b_(x, cos, sin, None)
    torch.testing.assert_close(ya, yb, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- LoRA


def test_lora_projections_used_when_rank_positive(tiny_model_cfg):
    """rank>0 swaps the targeted projections for LoRALinear (needs ttt/model/lora.py)."""
    lora_mod = pytest.importorskip("ttt.model.lora", reason="ttt/model/lora.py not written yet")
    cfg = replace(tiny_model_cfg, lora=LoRAConfig(rank=2, alpha=4.0, targets=("wq", "wv")))
    attn = SlidingWindowAttention(cfg, use_math_backend=False)
    assert isinstance(attn.wq, lora_mod.LoRALinear)
    assert isinstance(attn.wv, lora_mod.LoRALinear)
    assert isinstance(attn.wk, torch.nn.Linear) and not isinstance(attn.wk, lora_mod.LoRALinear)
    assert isinstance(attn.wo, torch.nn.Linear) and not isinstance(attn.wo, lora_mod.LoRALinear)
    x = torch.randn(1, 4, cfg.hidden_size)
    cos, sin = build_rope_cache(cfg.head_dim, 4, cfg.rope)
    y, _ = attn(x, cos, sin, None)
    assert y.shape == x.shape
