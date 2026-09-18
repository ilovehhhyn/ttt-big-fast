"""Tests for ttt.model.lora.LoRALinear (Task 3a).

Written before the implementation: every test here pins a property the outer
loop depends on (zero-init identity, rsLoRA scaling, merge/forward agreement,
gradient flow to both factors, and rank-0 being a hard error).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from ttt.config import LoRAConfig
from ttt.model.lora import LoRALinear


def test_zero_init_is_identity() -> None:
    """B = 0 at init, so the adapter term is exactly zero and the layer is the base linear.

    This must be exact, not approximate: the outer loop starts from the
    pretrained model, and any perturbation at step 0 is a silent bug.
    """
    torch.manual_seed(0)
    base = nn.Linear(16, 8, bias=False)
    cfg = LoRAConfig(rank=4, alpha=8.0)
    layer = LoRALinear.from_linear(base, cfg)

    x = torch.randn(3, 5, 16)
    assert torch.equal(layer(x), base(x))
    assert torch.equal(layer.lora_B, torch.zeros_like(layer.lora_B))
    # A must NOT be zero, otherwise B would never receive gradient.
    assert layer.lora_A.abs().sum() > 0.0


def test_zero_init_is_identity_with_bias() -> None:
    torch.manual_seed(0)
    base = nn.Linear(16, 8, bias=True)
    with torch.no_grad():
        base.bias.normal_()
    layer = LoRALinear.from_linear(base, LoRAConfig(rank=4, alpha=8.0))
    x = torch.randn(7, 16)
    assert layer.bias is not None
    assert torch.equal(layer(x), base(x))


def test_from_linear_does_not_share_storage() -> None:
    base = nn.Linear(6, 4, bias=True)
    layer = LoRALinear.from_linear(base, LoRAConfig(rank=2, alpha=4.0))
    with torch.no_grad():
        base.weight.add_(1.0)
        base.bias.add_(1.0)
    assert not torch.equal(layer.weight, base.weight)
    assert not torch.equal(layer.bias, base.bias)


def test_rslora_and_classic_scale() -> None:
    """rsLoRA (arXiv:2312.03732 Thm 3.2): alpha/sqrt(r), not alpha/r."""
    rs = LoRALinear(8, 8, LoRAConfig(rank=64, alpha=16.0, scaling="rslora"))
    classic = LoRALinear(8, 8, LoRAConfig(rank=64, alpha=16.0, scaling="classic"))
    assert rs.scale == pytest.approx(2.0)  # 16 / sqrt(64) = 16/8
    assert classic.scale == pytest.approx(0.25)  # 16 / 64


def test_merged_weight_matches_forward() -> None:
    torch.manual_seed(0)
    cfg = LoRAConfig(rank=4, alpha=8.0)
    layer = LoRALinear(16, 8, cfg)
    with torch.no_grad():
        layer.lora_B.normal_(0.0, 0.5)

    x = torch.randn(11, 16)
    merged = layer.merged_weight()
    assert merged.shape == (8, 16)
    torch.testing.assert_close(x @ merged.T, layer(x), rtol=1e-5, atol=1e-5)


def test_merged_weight_matches_forward_with_bias() -> None:
    torch.manual_seed(0)
    layer = LoRALinear(16, 8, LoRAConfig(rank=4, alpha=8.0), bias=True)
    with torch.no_grad():
        layer.lora_B.normal_(0.0, 0.5)
        layer.bias.normal_()
    x = torch.randn(11, 16)
    torch.testing.assert_close(
        x @ layer.merged_weight().T + layer.bias, layer(x), rtol=1e-5, atol=1e-5
    )


def test_gradients_reach_A_and_B() -> None:
    """Both factors are trainable and receive non-zero gradient.

    Note the one-step asymmetry that zero-init forces:
        dL/dB = scale * dL/dy^T (x A^T)      -- non-zero at init
        dL/dA = scale * B^T dL/dy^T x        -- ZERO at init, because B = 0
    So A only starts moving once the first outer step has made B non-zero.
    That is inherent to zero-init LoRA, not a wiring bug; we assert both the
    init behaviour and the steady-state behaviour.
    """
    torch.manual_seed(0)
    layer = LoRALinear(16, 8, LoRAConfig(rank=4, alpha=8.0))
    x = torch.randn(5, 16)
    layer(x).pow(2).sum().backward()

    assert layer.lora_B.grad is not None and layer.lora_B.grad.abs().sum() > 0.0
    # A is wired into the graph but its gradient is exactly zero while B == 0.
    assert layer.lora_A.grad is not None
    assert layer.lora_A.grad.abs().sum() == 0.0

    # After B moves off zero (i.e. after one outer step), both get gradient.
    layer.lora_A.grad = None
    layer.lora_B.grad = None
    with torch.no_grad():
        layer.lora_B.normal_(0.0, 0.5)
    layer(x).pow(2).sum().backward()

    assert layer.lora_A.grad is not None and layer.lora_A.grad.abs().sum() > 0.0
    assert layer.lora_B.grad is not None and layer.lora_B.grad.abs().sum() > 0.0


def test_rank_zero_disabled() -> None:
    """rank=0 is a configuration error, not a no-op adapter.

    A rank-0 LoRALinear would allocate empty factors and silently behave as a
    plain nn.Linear; callers that do not want an adapter must use nn.Linear.
    """
    with pytest.raises(ValueError, match="rank"):
        LoRALinear(8, 8, LoRAConfig(rank=0))
    with pytest.raises(ValueError, match="rank"):
        LoRALinear.from_linear(nn.Linear(8, 8, bias=False), LoRAConfig(rank=0))


def test_shapes_and_attributes() -> None:
    cfg = LoRAConfig(rank=4, alpha=8.0)
    layer = LoRALinear(16, 8, cfg)
    assert layer.weight.shape == (8, 16)
    assert layer.lora_A.shape == (4, 16)
    assert layer.lora_B.shape == (8, 4)
    assert layer.bias is None
    assert layer.cfg is cfg
    names = dict(layer.named_parameters())
    assert set(names) == {"weight", "lora_A", "lora_B"}
