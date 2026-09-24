"""Tests for the inner (test-time) optimizer.

Everything here runs in float64: the outer loop differentiates through these
updates, so we care about exact algebra, not speed.

The three properties that matter and are easy to lose:
  * purity (no in-place, no .data, no no_grad) -- the chunk chain
    W_0 -> W_1 -> ... -> W_N must stay in the autograd graph;
  * double differentiability -- grad-of-grad must flow through the update;
  * unit consistency -- `lr_rms` is the per-element RMS of the update for all
    three optimizers, so switching optimizer does not silently rescale the
    inner learning rate.
"""

from __future__ import annotations

import pytest
import torch

from ttt.config import InnerConfig
from ttt.optim.inner import (
    DifferentiableAdamW,
    MuonNoMomentum,
    NormalizedSGD,
    build_inner_optimizer,
    newton_schulz5,
)

DT = torch.float64


def _tensors(seed: int, *, requires_grad: bool = False) -> dict[str, torch.Tensor]:
    """A two-tensor pytree: one matrix (Muon path) and one vector (fallback path)."""
    gen = torch.Generator().manual_seed(seed)
    return {
        "a": torch.randn(6, 8, generator=gen, dtype=DT, requires_grad=requires_grad),
        "b": torch.randn(16, generator=gen, dtype=DT, requires_grad=requires_grad),
    }


def _clone(d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in d.items()}


def _rms(t: torch.Tensor) -> float:
    return t.pow(2).mean().sqrt().item()


# --------------------------------------------------------------------------- #
# 1-2. NormalizedSGD: units and floor
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("scope", ["tensor", "global"])
def test_normalized_sgd_rms_is_lr(scope: str) -> None:
    """||update|| = lr*sqrt(n) => per-element RMS = lr, exactly the point of the sqrt(numel) factor.

    The residual error is lr * eps_norm/||g||, i.e. ~1e-10 here, far below 1e-6.
    """
    cfg = InnerConfig(optimizer="normalized_sgd", lr_rms=1e-2, norm_scope=scope)
    opt = build_inner_optimizer(cfg)
    assert isinstance(opt, NormalizedSGD)

    fast, grads = _tensors(0), _tensors(1)
    state = opt.init_state(fast)
    new_fast, _ = opt.step(fast, grads, state)
    deltas = {k: new_fast[k] - fast[k] for k in fast}

    if scope == "tensor":
        # each tensor independently has per-element RMS lr_rms
        for k, d in deltas.items():
            assert abs(_rms(d) - cfg.lr_rms) < 1e-6, k
    else:
        # only the concatenation has RMS lr_rms; individual tensors do not
        flat = torch.cat([d.reshape(-1) for d in deltas.values()])
        assert abs(_rms(flat) - cfg.lr_rms) < 1e-6


@pytest.mark.parametrize("scope", ["tensor", "global"])
def test_normalized_sgd_floor(scope: str) -> None:
    """||g|| < eps_norm must give an exactly zero step, not a 1/||g|| explosion."""
    cfg = InnerConfig(optimizer="normalized_sgd", lr_rms=1e-2, eps_norm=1e-6, norm_scope=scope)
    opt = build_inner_optimizer(cfg)

    fast = _tensors(0)
    grads = {k: torch.full_like(v, 1e-12) for k, v in fast.items()}
    assert all(g.norm().item() < cfg.eps_norm for g in grads.values())

    new_fast, _ = opt.step(fast, grads, opt.init_state(fast))
    for k in fast:
        assert torch.equal(new_fast[k], fast[k]), k


# --------------------------------------------------------------------------- #
# 3. Purity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["none", "normalized_sgd", "adamw", "muon"])
def test_purity(name: str) -> None:
    """step() must not mutate fast, grads, or state (their tensors or their dicts)."""
    cfg = InnerConfig(optimizer=name, lr_rms=1e-2, warm_start=True)
    opt = build_inner_optimizer(cfg)

    fast, grads, first = _tensors(0), _tensors(1), _tensors(2)
    state = opt.init_state(fast, first)

    fast_before, grads_before, first_before = _clone(fast), _clone(grads), _clone(first)
    state_keys_before = set(state)
    moments_before = {
        which: _clone(state[which]) for which in ("m", "v") if which in state
    }
    t_before = state.get("t")

    new_fast, new_state = opt.step(fast, grads, state)

    for k in fast:
        assert torch.equal(fast[k], fast_before[k]), f"fast[{k}] mutated"
        assert torch.equal(grads[k], grads_before[k]), f"grads[{k}] mutated"
        assert torch.equal(first[k], first_before[k]), f"first_grad[{k}] mutated"
    assert set(state) == state_keys_before
    for which, snap in moments_before.items():
        for k, v in snap.items():
            assert torch.equal(state[which][k], v), f"state[{which}][{k}] mutated"
    assert state.get("t") == t_before
    assert new_fast is not fast
    assert new_state is not state


def test_noop_returns_fast_unchanged() -> None:
    opt = build_inner_optimizer(InnerConfig(optimizer="none"))
    fast, grads = _tensors(0), _tensors(1)
    new_fast, new_state = opt.step(fast, grads, opt.init_state(fast))
    assert new_fast is not fast
    for k in fast:
        assert new_fast[k] is fast[k]
    assert new_state == {}


# --------------------------------------------------------------------------- #
# 4. The critical one: grad-of-grad flows through the update
# --------------------------------------------------------------------------- #


def _second_order_probe(cfg: InnerConfig, shape: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Build g = dL_inner/dW with create_graph, step, then differentiate the step.

    Returns (d loss / d g, d loss / d x). The second one is the real meta-gradient
    shape: it only reaches x by differentiating *through* the optimizer step and
    then through the gradient computation that produced g.
    """
    x = torch.randn(shape, dtype=DT, requires_grad=True)
    w = torch.randn(shape, dtype=DT, requires_grad=True)

    inner_loss = 0.5 * (w * x).pow(2).sum() + (w * x.sin()).sum()
    (g,) = torch.autograd.grad(inner_loss, w, create_graph=True)

    opt = build_inner_optimizer(cfg)
    first = {"w": torch.randn(shape, dtype=DT)}
    state = opt.init_state({"w": w}, first)
    new_fast, _ = opt.step({"w": w}, {"w": g}, state)

    c = torch.randn(shape, dtype=DT)
    loss = (new_fast["w"] * c).sum() + new_fast["w"].pow(2).sum()

    (dg,) = torch.autograd.grad(loss, g, create_graph=True, retain_graph=True)
    (dx,) = torch.autograd.grad(loss, x, retain_graph=True)
    return dg, dx


@pytest.mark.parametrize("name, ns_dtype", [("normalized_sgd", "float32"), ("adamw", "float32"), ("muon", "float32"), ("muon", "bfloat16")])
def test_second_order_flows(name: str, ns_dtype: str) -> None:
    cfg = InnerConfig(optimizer=name, lr_rms=1e-2, warm_start=True, ns_dtype=ns_dtype)
    dg, dx = _second_order_probe(cfg, (6, 8))

    assert torch.isfinite(dg).all(), f"{name}: non-finite d loss/d g"
    assert dg.abs().max().item() > 0.0, f"{name}: d loss/d g is identically zero"
    assert torch.isfinite(dx).all(), f"{name}: non-finite second-order grad"
    assert dx.abs().max().item() > 0.0, f"{name}: second-order grad is identically zero"


# --------------------------------------------------------------------------- #
# 5-6. DifferentiableAdamW
# --------------------------------------------------------------------------- #


def test_adamw_warm_start_no_singularity() -> None:
    """Zeros and 1e-12 entries must not produce NaN/inf in the update or its gradient."""
    cfg = InnerConfig(optimizer="adamw", lr_rms=1e-3, warm_start=True, eps=1e-8)
    opt = build_inner_optimizer(cfg)
    assert isinstance(opt, DifferentiableAdamW)

    w = torch.randn(3, 4, dtype=DT)
    nasty = torch.tensor(
        [
            [0.0, 1e-12, -1e-12, 0.0],
            [1e-12, 0.0, 1.0, -1.0],
            [0.0, 0.0, 1e-12, 3.0],
        ],
        dtype=DT,
        requires_grad=True,
    )
    state = opt.init_state({"w": w}, {"w": nasty})
    new_fast, _ = opt.step({"w": w}, {"w": nasty}, state)

    delta = new_fast["w"] - w
    assert torch.isfinite(delta).all()
    (grad,) = torch.autograd.grad(delta.pow(2).sum(), nasty)
    assert torch.isfinite(grad).all()


def test_adamw_matches_torch_adamw_first_steps() -> None:
    """Cold start must reproduce torch.optim.AdamW(weight_decay=0).

    Known, deliberate discrepancy: torch divides by sqrt(v_hat) + eps, we divide by
    sqrt(v_hat + eps**2) -- ours has a finite derivative at v = 0, which the outer
    loop needs. The two differ by O(eps) in the denominator, so with eps = 1e-12 the
    per-step difference is ~lr*1e-12 and the comparison at 1e-6 is meaningful rather
    than vacuous. At the default eps = 1e-8 the two would differ by ~lr*1e-8 per step.
    """
    cfg = InnerConfig(optimizer="adamw", lr_rms=1e-3, warm_start=False, eps=1e-12, beta1=0.9, beta2=0.99)
    opt = build_inner_optimizer(cfg)

    w0 = torch.randn(5, 7, dtype=DT)
    fast = {"w": w0.clone()}
    state = opt.init_state(fast)

    p = torch.nn.Parameter(w0.clone())
    ref = torch.optim.AdamW([p], lr=cfg.lr_rms, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps, weight_decay=0.0)

    gen = torch.Generator().manual_seed(7)
    for _ in range(4):
        g = torch.randn(5, 7, generator=gen, dtype=DT)
        fast, state = opt.step(fast, {"w": g}, state)
        p.grad = g.clone()
        ref.step()

    assert torch.allclose(fast["w"], p.detach(), atol=1e-6, rtol=0.0), (
        (fast["w"] - p.detach()).abs().max().item()
    )


# --------------------------------------------------------------------------- #
# 7-8. Muon
# --------------------------------------------------------------------------- #


def test_newton_schulz_orthogonalises() -> None:
    """Singular values collapse from a wide spread onto a narrow band around 1.

    DEVIATION from the 0.3 bound in the task spec: 0.35. Jordan's (a, b, c) =
    (3.4445, -4.7750, 2.0315) are tuned to maximise the slope at 0 rather than to
    converge to 1, so 5 steps land the spectrum in ~[0.682, 1.134] -- a max deviation
    of 0.318, measured as stable to 3 decimals across seeds (it is a property of the
    quintic, not of the draw). 0.3 would fail for every input; 0.35 is the tightest
    bound the reference NS5 actually satisfies. The input spectrum here spans
    [0.037, 0.213] after Frobenius normalisation, so this is a ~4.8x -> ~1.7x
    condition-number reduction, which is what Muon needs.
    """
    gen = torch.Generator().manual_seed(3)
    g = torch.randn(64, 128, generator=gen, dtype=DT)
    s = torch.linalg.svdvals(newton_schulz5(g))
    assert s.numel() == 64
    assert (s - 1.0).abs().max().item() < 0.35, s
    # and it really did orthogonalise: the raw input is far from the band
    s_raw = torch.linalg.svdvals(g / g.reshape(-1).norm())
    assert (s_raw - 1.0).abs().max().item() > 0.7, s_raw


def test_newton_schulz_bf16_iteration_matches_fp32_within_5_percent() -> None:
    """Oracle: the fp32 iteration on the same input.

    Invariant: the bf16 iteration returns the input dtype and lands within 5% relative
    Frobenius error of the fp32 result, with singular values in the same band as test 7.
    Witness: measured 0.0326 on this 64 x 128 draw and 0.0191 on the real 2048 x 8192 fast
    matrix shape (2026-09-23); 0.05 is the tightest round bound above both.
    """
    gen = torch.Generator().manual_seed(3)
    g = torch.randn(64, 128, generator=gen, dtype=torch.float32)

    reference = newton_schulz5(g, iteration_dtype="float32")
    low = newton_schulz5(g, iteration_dtype="bfloat16")

    assert low.dtype == torch.float32, low.dtype
    relative = ((low - reference).norm() / reference.norm()).item()
    assert relative < 0.05, relative
    # Control: the bf16 path really ran; a silent fp32 fallback would be bit-identical.
    assert relative > 1e-4, relative
    s = torch.linalg.svdvals(low)
    assert (s - 1.0).abs().max().item() < 0.35, s


def test_newton_schulz_rejects_an_unknown_iteration_dtype() -> None:
    g = torch.randn(4, 8, dtype=DT)
    with pytest.raises(AssertionError, match="iteration_dtype"):
        newton_schulz5(g, iteration_dtype="float16")  # type: ignore[arg-type]  # the rejected case


def test_inner_config_rejects_ns_dtype_without_muon() -> None:
    with pytest.raises(AssertionError, match="ns_dtype.*muon"):
        InnerConfig(optimizer="normalized_sgd", lr_rms=1e-2, ns_dtype="bfloat16")


def test_muon_bf16_step_keeps_the_fast_weight_dtype() -> None:
    cfg = InnerConfig(optimizer="muon", lr_rms=1e-2, ns_dtype="bfloat16")
    opt = build_inner_optimizer(cfg)
    fast = {"a": torch.randn(6, 8, dtype=torch.float32)}
    grads = {"a": torch.randn(6, 8, dtype=torch.float32)}

    new_fast, _ = opt.step(fast, grads, opt.init_state(fast))

    assert new_fast["a"].dtype == torch.float32
    assert abs(_rms(new_fast["a"] - fast["a"]) - cfg.lr_rms) < 0.15 * cfg.lr_rms


def test_muon_rms_is_lr() -> None:
    """15% because NS5 is a 5-step approximation, not an exact polar factor.

    NS5(g) has singular values in roughly [0.7, 1.3] (test 7), so its per-element RMS
    sqrt(mean(s^2)/max(m,n)) sits within ~15% of 1/sqrt(max(m,n)); the sqrt(max(m,n))
    factor then puts the update RMS within ~15% of lr_rms. Tightening this would mean
    asserting a property NS5 does not have.
    """
    cfg = InnerConfig(optimizer="muon", lr_rms=1e-2)
    opt = build_inner_optimizer(cfg)
    assert isinstance(opt, MuonNoMomentum)

    fast, grads = _tensors(0), _tensors(1)
    # a strongly rectangular tensor, so that the sqrt(max(m,n)) factor is pinned:
    # sqrt(max)/sqrt(min) = 2 here, so using min (or numel, or nothing) misses by >=100%
    # and cannot hide inside the 15% band.
    gen = torch.Generator().manual_seed(11)
    fast["r"] = torch.randn(16, 64, generator=gen, dtype=DT)
    grads["r"] = torch.randn(16, 64, generator=gen, dtype=DT)
    new_fast, _ = opt.step(fast, grads, opt.init_state(fast))

    for k in ("a", "r"):
        d2 = new_fast[k] - fast[k]
        assert abs(_rms(d2) - cfg.lr_rms) < 0.15 * cfg.lr_rms, (k, _rms(d2))

    # the same holds with the tall orientation (NS5 transposes internally)
    tall = {"t": fast["r"].mT.contiguous()}
    tall_g = {"t": grads["r"].mT.contiguous()}
    nf, _ = opt.step(tall, tall_g, opt.init_state(tall))
    assert abs(_rms(nf["t"] - tall["t"]) - cfg.lr_rms) < 0.15 * cfg.lr_rms

    # 1-D tensors fall back to the NormalizedSGD rule, which is exact
    d1 = new_fast["b"] - fast["b"]
    assert abs(_rms(d1) - cfg.lr_rms) < 1e-6


# --------------------------------------------------------------------------- #
# 9-10. Learned per-tensor LR and key hygiene
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["normalized_sgd", "adamw", "muon"])
def test_lr_mult_is_differentiable(name: str) -> None:
    """lr_mult is exp(inner_lr_log), a meta-learned slow parameter: never detach it."""
    cfg = InnerConfig(optimizer=name, lr_rms=1e-2, warm_start=True)
    opt = build_inner_optimizer(cfg)

    fast, grads, first = _tensors(0), _tensors(1), _tensors(2)
    log_lr = {k: torch.zeros((), dtype=DT, requires_grad=True) for k in fast}
    lr_mult = {k: v.exp() for k, v in log_lr.items()}

    state = opt.init_state(fast, first)
    new_fast, _ = opt.step(fast, grads, state, lr_scale=torch.tensor(0.5, dtype=DT), lr_mult=lr_mult)
    sum(v.pow(2).sum() for v in new_fast.values()).backward()

    for k, t in log_lr.items():
        assert t.grad is not None, k
        assert torch.isfinite(t.grad).all(), k
        assert t.grad.abs().item() > 0.0, k


@pytest.mark.parametrize("name", ["normalized_sgd", "adamw", "muon"])
def test_keys_must_match(name: str) -> None:
    cfg = InnerConfig(optimizer=name, lr_rms=1e-2, warm_start=False)
    opt = build_inner_optimizer(cfg)
    fast = _tensors(0)
    state = opt.init_state(fast)

    missing = {"a": torch.zeros(6, 8, dtype=DT)}
    with pytest.raises(AssertionError):
        opt.step(fast, missing, state)

    extra = {**_tensors(1), "c": torch.zeros(3, dtype=DT)}
    with pytest.raises(AssertionError):
        opt.step(fast, extra, state)

    with pytest.raises(AssertionError):
        opt.step(fast, _tensors(1), state, lr_mult={"zzz": torch.ones((), dtype=DT)})
