"""TransformerBlock (and SwiGLU MLP) tests, including the second-order path."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from ttt.config import LoRAConfig, ModelConfig
from ttt.model.attention import KVCache
from ttt.model.block import TransformerBlock
from ttt.model.mlp import SwiGLUMLP
from ttt.model.rope import build_rope_cache


def _no_lora(cfg: ModelConfig) -> ModelConfig:
    return replace(cfg, lora=LoRAConfig(rank=0))


# --------------------------------------------------------------------------- MLP


def test_swiglu_formula():
    """y = w2(silu(w1(x)) * w3(x))."""
    torch.manual_seed(0)
    mlp = SwiGLUMLP(8, 16, LoRAConfig(rank=0))
    x = torch.randn(2, 5, 8)
    want = mlp.w2(F.silu(mlp.w1(x)) * mlp.w3(x))
    torch.testing.assert_close(mlp(x), want)
    assert mlp.w1.weight.shape == (16, 8)
    assert mlp.w3.weight.shape == (16, 8)
    assert mlp.w2.weight.shape == (8, 16)
    assert mlp.w1.bias is None


def test_swiglu_lora_targets():
    lora_mod = pytest.importorskip("ttt.model.lora", reason="ttt/model/lora.py not written yet")
    mlp = SwiGLUMLP(8, 16, LoRAConfig(rank=2, alpha=4.0, targets=("w1", "w2")))
    assert isinstance(mlp.w1, lora_mod.LoRALinear)
    assert isinstance(mlp.w2, lora_mod.LoRALinear)
    assert not isinstance(mlp.w3, lora_mod.LoRALinear)
    assert mlp(torch.randn(1, 3, 8)).shape == (1, 3, 8)


# --------------------------------------------------------------------------- block


def test_block_submodule_names(tiny_model_cfg):
    cfg = _no_lora(tiny_model_cfg)
    block = TransformerBlock(cfg, use_math_backend=False)
    names = dict(block.named_children())
    assert set(names) == {"seq_norm", "attn", "ffn_norm", "mlp"}
    assert isinstance(block.seq_norm, torch.nn.RMSNorm)
    assert isinstance(block.ffn_norm, torch.nn.RMSNorm)
    assert block.seq_norm.eps == cfg.rms_norm_eps
    assert isinstance(block.mlp, SwiGLUMLP)


def test_block_is_prenorm_residual(tiny_model_cfg):
    """x = x + attn(seq_norm(x)); x = x + mlp(ffn_norm(x)). No post-norm anywhere."""
    cfg = _no_lora(tiny_model_cfg)
    torch.manual_seed(0)
    block = TransformerBlock(cfg, use_math_backend=True)
    x = torch.randn(2, 12, cfg.hidden_size)
    cos, sin = build_rope_cache(cfg.head_dim, 12, cfg.rope)
    got, _ = block(x, cos, sin, None)
    h, _ = block.attn(block.seq_norm(x), cos, sin, None)
    h = x + h
    want = h + block.mlp(block.ffn_norm(h))
    torch.testing.assert_close(got, want)


def test_block_shapes_and_grad(tiny_model_cfg):
    cfg = _no_lora(tiny_model_cfg)
    block = TransformerBlock(cfg, use_math_backend=True)
    x = torch.randn(2, 12, cfg.hidden_size)
    cos, sin = build_rope_cache(cfg.head_dim, 12, cfg.rope)
    y, cache = block(x, cos, sin, None)
    assert y.shape == x.shape
    assert cache is None

    params = [block.mlp.w1.weight, block.mlp.w2.weight, block.mlp.w3.weight]
    grads = torch.autograd.grad(y.sum(), params)
    for g, p in zip(grads, params, strict=True):
        assert g.shape == p.shape
        assert torch.isfinite(g).all()
        assert g.abs().sum() > 0


def test_double_backward_math_backend(tiny_model_cfg):
    """Second-order path: grad of a grad must flow (the TTT outer loop needs this)."""
    cfg = _no_lora(tiny_model_cfg)
    block = TransformerBlock(cfg, use_math_backend=True)
    x = torch.randn(1, 8, cfg.hidden_size)
    cos, sin = build_rope_cache(cfg.head_dim, 8, cfg.rope)
    y, _ = block(x, cos, sin, None)
    loss = (y**2).sum()

    mlp_w = block.mlp.w2.weight
    g = torch.autograd.grad(loss, mlp_w, create_graph=True)[0]
    assert g.requires_grad, "first-order grad must stay in the graph"
    g2 = torch.autograd.grad(g.sum(), block.attn.wq.weight)[0]
    assert torch.isfinite(g2).all()
    assert g2.abs().sum() > 0


def test_block_chunked_equals_full(tiny_model_cfg):
    cfg = _no_lora(tiny_model_cfg)
    block = TransformerBlock(cfg, use_math_backend=True).double()
    seq, b = 16, 2
    x = torch.randn(b, seq, cfg.hidden_size, dtype=torch.float64)
    cos, sin = build_rope_cache(cfg.head_dim, seq, cfg.rope, dtype=torch.float64)
    full, _ = block(x, cos, sin, None)

    cache = KVCache.empty(b, cfg.window_size, cfg.num_kv_heads, cfg.head_dim, device=x.device, dtype=x.dtype)
    outs = []
    for start in range(0, seq, cfg.chunk_size):
        stop = start + cfg.chunk_size
        y, cache = block(x[:, start:stop], cos[start:stop], sin[start:stop], cache)
        outs.append(y)
    torch.testing.assert_close(torch.cat(outs, dim=1), full, atol=1e-5, rtol=1e-5)
