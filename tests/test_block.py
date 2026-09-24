"""TransformerBlock (and SwiGLU MLP) tests, including the second-order path."""

import math
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from ttt.config import LoRAConfig, ModelConfig
from ttt.model.attention import KVCache
from ttt.model.block import TransformerBlock
from ttt.model.mlp import SwiGLUMLP
from ttt.model.rope import build_rope_cache
from ttt.model.token_rate import TOKEN_RATE_INIT, TOKEN_RATE_OFFSET, TokenRate, scale_gradient


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


# --------------------------------------------------------------------------- per-token rates


def test_scale_gradient_weights_each_token_write_and_leaves_the_value_unchanged():
    """Invariant: y = W k_t is unchanged in value; dL/dW = sum_t eta_t d_t k_t^T exactly.
    Witness: a one-hot eta gives the rank-one outer product of token t0 alone."""
    torch.manual_seed(0)
    w = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    k = torch.randn(7, 3, dtype=torch.float64)
    c = torch.randn(7, 5, dtype=torch.float64)
    eta = torch.zeros(7, 1, dtype=torch.float64)
    eta[2] = 3.0

    y = scale_gradient(k @ w.T, eta)
    (g,) = torch.autograd.grad((y * c).sum(), w)

    torch.testing.assert_close(y, k @ w.T, rtol=0.0, atol=0.0)
    want = 3.0 * torch.outer(c[2], k[2])
    torch.testing.assert_close(g, want, rtol=0.0, atol=1e-12)


def test_scale_gradient_is_differentiable_in_eta_through_the_gradient():
    """The meta-gradient reaches eta only through d(dL/dW)/d(eta); the forward value
    carries none, so a first-order gradient in eta is zero and a second-order one is not."""
    torch.manual_seed(0)
    w = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    k = torch.randn(6, 3, dtype=torch.float64)
    eta = torch.full((6, 1), 1.5, dtype=torch.float64, requires_grad=True)

    y = scale_gradient(k @ w.T, eta)
    (first,) = torch.autograd.grad(y.sum(), eta, retain_graph=True, allow_unused=True)
    (g,) = torch.autograd.grad(y.pow(2).sum(), w, create_graph=True)
    (second,) = torch.autograd.grad(g.pow(2).sum(), eta)

    assert first is None or first.abs().max().item() == 0.0
    assert torch.isfinite(second).all() and second.abs().max().item() > 0.0


def test_token_rate_starts_at_exactly_one_for_every_token():
    rate = TokenRate(hidden_size=8)
    x = torch.randn(2, 5, 8)
    torch.testing.assert_close(rate(x), torch.full((2, 5, 1), TOKEN_RATE_INIT), rtol=0.0, atol=1e-6)
    assert rate.linear.weight.abs().max().item() == 0.0
    assert rate.linear.bias.item() == 0.0


def test_block_with_token_rates_matches_block_without_at_init(tiny_model_cfg):
    """At init eta = 1 everywhere, so values and fast-weight gradients are identical."""
    cfg = _no_lora(tiny_model_cfg)
    torch.manual_seed(0)
    plain = TransformerBlock(cfg, use_math_backend=True, is_fast_block=True)
    rated = TransformerBlock(replace(cfg, token_rates=True), use_math_backend=True, is_fast_block=True)
    rated.load_state_dict(plain.state_dict(), strict=False)
    x = torch.randn(1, 8, cfg.hidden_size)
    cos, sin = build_rope_cache(cfg.head_dim, 8, cfg.rope)

    y_plain, _ = plain(x, cos, sin, None)
    y_rated, _ = rated(x, cos, sin, None)
    g_plain = torch.autograd.grad((y_plain**2).sum(), plain.mlp.w2.weight)[0]
    g_rated = torch.autograd.grad((y_rated**2).sum(), rated.mlp.w2.weight)[0]

    assert rated.token_rate is not None and not hasattr(plain, "token_rate")
    torch.testing.assert_close(y_rated, y_plain, rtol=0.0, atol=0.0)
    torch.testing.assert_close(g_rated, g_plain, rtol=0.0, atol=0.0)


def test_block_token_rate_of_two_doubles_every_fast_weight_gradient(tiny_model_cfg):
    """The write of each token is weighted by its rate: a constant rate c scales the
    gradient of every fast matrix by c exactly, and second order reaches the rate."""
    cfg = replace(_no_lora(tiny_model_cfg), token_rates=True)
    torch.manual_seed(0)
    block = TransformerBlock(cfg, use_math_backend=True, is_fast_block=True).double()
    x = torch.randn(1, 8, cfg.hidden_size, dtype=torch.float64)
    cos, sin = build_rope_cache(cfg.head_dim, 8, cfg.rope)
    cos, sin = cos.double(), sin.double()
    params = [block.mlp.w1.weight, block.mlp.w2.weight, block.mlp.w3.weight]

    y1, _ = block(x, cos, sin, None)
    g1 = torch.autograd.grad((y1**2).sum(), params)
    with torch.no_grad():
        block.token_rate.linear.bias.fill_(math.log(math.expm1(2.0)) - TOKEN_RATE_OFFSET)  # softplus(b + offset) = 2
    y2, _ = block(x, cos, sin, None)
    g2 = torch.autograd.grad((y2**2).sum(), params, create_graph=True)
    meta = torch.autograd.grad(sum(g.pow(2).sum() for g in g2), block.token_rate.linear.weight)[0]

    torch.testing.assert_close(y2, y1, rtol=0.0, atol=0.0)
    for a, b in zip(g1, g2, strict=True):
        torch.testing.assert_close(b, 2.0 * a, rtol=1e-12, atol=1e-12)
    assert torch.isfinite(meta).all() and meta.abs().max().item() > 0.0
