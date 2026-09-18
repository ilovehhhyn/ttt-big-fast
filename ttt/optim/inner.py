"""Inner (test-time) optimizers.

Once per chunk the fast weights W take a single step, and the outer loop then
differentiates through the whole chain W_0 -> W_1 -> ... -> W_N. Every function
in this module is therefore written under two hard constraints:

  * FUNCTIONAL -- no in-place ops, no `.data`, no `torch.no_grad()`. `step` takes
    a dict of tensors and returns a NEW dict; its inputs are never touched.
  * DIFFERENTIABLE TWICE -- the update is a smooth function of the gradient, so
    `torch.autograd.grad(..., create_graph=True)` flows through it and the
    resulting meta-gradient is itself differentiable.

The TTT-E2E reference (github.com/test-time-training/e2e, `ttt/optimizers.py`)
uses `optax.chain(clip_by_global_norm(1.0), sgd(lr=1.0, momentum=None))`, i.e.
the gradient is normalised only when its norm exceeds 1. We go further and
normalise strictly, every step, so the update has a known scale regardless of
where in the sequence the chunk sits.

UNIT CONVENTION. All three optimizers are scaled so that `cfg.lr_rms` is the
per-element RMS of the update. Switching optimizer therefore does not silently
rescale the inner learning rate, and a single warmup schedule applies to all.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor

from ttt.config import InnerConfig

__all__ = [
    "InnerOptimizer",
    "NoOpInnerOptimizer",
    "NormalizedSGD",
    "DifferentiableAdamW",
    "MuonNoMomentum",
    "build_inner_optimizer",
    "newton_schulz5",
]


def _check_keys(fast: dict[str, Tensor], grads: dict[str, Tensor], lr_mult: dict[str, Tensor] | None) -> None:
    """The gradient pytree must match the fast-weight pytree exactly."""
    assert set(grads) == set(fast), (
        f"grads keys must match fast keys exactly; "
        f"missing={sorted(set(fast) - set(grads))} extra={sorted(set(grads) - set(fast))}"
    )
    if lr_mult is not None:
        assert set(lr_mult) <= set(fast), f"lr_mult has unknown keys: {sorted(set(lr_mult) - set(fast))}"


def _lr(cfg: InnerConfig, key: str, lr_scale: Tensor | float, lr_mult: dict[str, Tensor] | None) -> Tensor | float:
    """cfg.lr_rms * lr_scale * lr_mult[key].

    `lr_scale` is the outer-step warmup (e2e ramps 0.1x -> 1x over the first 10%
    of training). `lr_mult` is the learned per-tensor inner LR, arriving as
    exp(inner_lr_log); it is meta-learned, so it is NEVER detached.
    """
    lr = cfg.lr_rms * lr_scale
    if lr_mult is not None and key in lr_mult:
        lr = lr * lr_mult[key]
    return lr


class InnerOptimizer(ABC):
    def __init__(self, cfg: InnerConfig) -> None:
        self.cfg = cfg

    @property
    def is_noop(self) -> bool:
        """True if step() is the identity, so the caller may skip the inner gradient."""
        return False

    @property
    def needs_first_grad(self) -> bool:
        """True if init_state requires the first chunk's gradient (AdamW warm start).

        The inner-loop runner defers init_state until after the first backward when
        this is True, so the moments can be seeded with m_0 = g_1, v_0 = g_1^2.
        """
        return False

    @abstractmethod
    def init_state(self, fast: dict[str, Tensor], first_grad: dict[str, Tensor] | None = None) -> dict[str, Any]:
        ...

    @abstractmethod
    def step(
        self,
        fast: dict[str, Tensor],
        grads: dict[str, Tensor],
        state: dict[str, Any],
        *,
        lr_scale: Tensor | float = 1.0,
        lr_mult: dict[str, Tensor] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Any]]:
        """Return (new_fast, new_state). Pure: inputs are not mutated."""


class NoOpInnerOptimizer(InnerOptimizer):
    @property
    def is_noop(self) -> bool:
        return True

    """cfg.optimizer == 'none': the fast weights never move (ablation baseline)."""

    def init_state(self, fast: dict[str, Tensor], first_grad: dict[str, Tensor] | None = None) -> dict[str, Any]:
        return {}

    def step(
        self,
        fast: dict[str, Tensor],
        grads: dict[str, Tensor],
        state: dict[str, Any],
        *,
        lr_scale: Tensor | float = 1.0,
        lr_mult: dict[str, Tensor] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Any]]:
        _check_keys(fast, grads, lr_mult)
        return dict(fast), {}


class NormalizedSGD(InnerOptimizer):
    """Per-tensor:   W <- W - lr * m_t * sqrt(numel(W)) * g / (||g||_F + eps_norm)
       Global:       W <- W - lr * m_t * sqrt(total_numel) * g / (||g_all||_F + eps_norm)

    The sqrt(numel) factor makes `lr` the per-element RMS of the update:
        ||update|| = lr*sqrt(n)  =>  RMS = ||update||/sqrt(n) = lr.
    A floor is applied: if ||g|| < eps_norm the step is exactly zero (a late, well-fit chunk
    produces a vanishing gradient, and 1/||g|| would otherwise blow up the meta-gradient).
    State is empty.
    """

    def init_state(self, fast: dict[str, Tensor], first_grad: dict[str, Tensor] | None = None) -> dict[str, Any]:
        return {}

    def step(
        self,
        fast: dict[str, Tensor],
        grads: dict[str, Tensor],
        state: dict[str, Any],
        *,
        lr_scale: Tensor | float = 1.0,
        lr_mult: dict[str, Tensor] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Any]]:
        _check_keys(fast, grads, lr_mult)
        cfg = self.cfg
        keys = sorted(fast)

        if cfg.norm_scope == "global":
            # ||g_all||_F over the whole pytree, and one shared sqrt(total_numel).
            sq = torch.stack([grads[k].reshape(-1).pow(2).sum() for k in keys]).sum()
            global_norm = sq.sqrt()
            scale_n = math.sqrt(float(sum(fast[k].numel() for k in keys)))

        new_fast: dict[str, Tensor] = {}
        for k in keys:
            g, w = grads[k], fast[k]
            if cfg.norm_scope == "tensor":
                norm = g.reshape(-1).norm()
                n = math.sqrt(float(w.numel()))
            else:
                norm = global_norm
                n = scale_n
            new_fast[k] = w - _lr(cfg, k, lr_scale, lr_mult) * n * _normalized_direction(g, norm, cfg.eps_norm)
        return new_fast, {}


def _normalized_direction(g: Tensor, norm: Tensor, eps_norm: float) -> Tensor:
    """g / (||g|| + eps_norm), hard-floored to exactly zero when ||g|| < eps_norm.

    The gate is a 0/1 indicator, whose derivative is zero almost everywhere, so it
    costs nothing in the meta-gradient; the branch it kills is the one where
    d/dg [g/||g||] ~ 1/||g|| would dominate the outer gradient. `torch.where` is
    avoided in favour of a multiply because both branches here are finite anyway,
    and a multiply keeps the graph simple.
    """
    gate = (norm >= eps_norm).to(g.dtype)
    return gate * (g / (norm + eps_norm))


class DifferentiableAdamW(InnerOptimizer):
    """m <- b1*m + (1-b1)*g ; v <- b2*v + (1-b2)*g^2
       m_hat = m/(1-b1^t) ; v_hat = v/(1-b2^t)
       W <- W - lr * m_t * m_hat / sqrt(v_hat + eps^2)

    Note the denominator is sqrt(v_hat + eps**2), NOT sqrt(v_hat) + eps: the former has a
    finite derivative at v=0, which matters because the outer loop differentiates through it.
    When cfg.warm_start, init_state sets m = first_grad and v = first_grad**2 and t = 1, which
    removes the step-1 singularity (with a cold start, v=0 and the first update is
    g/(|g|+eps) whose derivative is eps/(|g|+eps)^2: ~0 for large |g| and ~1e8 for small |g|).
    weight decay is NOT applied to fast weights (they are reset every sequence).

    m_hat/sqrt(v_hat) is ~ +/-1 per element, so `lr` is again the per-element RMS of
    the update, matching NormalizedSGD and MuonNoMomentum.

    Bias correction and warm start (deviation, see module report): the 1-b^t factors
    correct an EMA started from ZERO. A warm-started state is already unbiased
    (m = g_1 estimates g, v = g_1^2 estimates g^2), so dividing it by 1-b^t would
    inflate the update by 1/sqrt(1-b2^t) -- e.g. 2.29x at t=2 with b=0.9 -- and break
    the RMS = lr convention. We therefore carry a `bias_correct` flag in the state and
    skip the correction on the warm path. `m`, `v` and `t` are exactly as specified.
    """

    @property
    def needs_first_grad(self) -> bool:
        return self.cfg.warm_start

    def init_state(self, fast: dict[str, Tensor], first_grad: dict[str, Tensor] | None = None) -> dict[str, Any]:
        if self.cfg.warm_start:
            assert first_grad is not None, "warm_start=True requires first_grad"
            assert set(first_grad) == set(fast), "first_grad keys must match fast keys exactly"
            return {
                "m": {k: first_grad[k] for k in fast},
                "v": {k: first_grad[k] * first_grad[k] for k in fast},
                "t": 1,
                "bias_correct": False,
            }
        return {
            "m": {k: torch.zeros_like(v) for k, v in fast.items()},
            "v": {k: torch.zeros_like(v) for k, v in fast.items()},
            "t": 0,
            "bias_correct": True,
        }

    def step(
        self,
        fast: dict[str, Tensor],
        grads: dict[str, Tensor],
        state: dict[str, Any],
        *,
        lr_scale: Tensor | float = 1.0,
        lr_mult: dict[str, Tensor] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Any]]:
        _check_keys(fast, grads, lr_mult)
        assert set(state["m"]) == set(fast) and set(state["v"]) == set(fast), "state keys must match fast keys"
        cfg = self.cfg
        b1, b2, eps = cfg.beta1, cfg.beta2, cfg.eps

        t = state["t"] + 1
        if state["bias_correct"]:
            bc1, bc2 = 1.0 - b1**t, 1.0 - b2**t
        else:
            bc1, bc2 = 1.0, 1.0

        new_m: dict[str, Tensor] = {}
        new_v: dict[str, Tensor] = {}
        new_fast: dict[str, Tensor] = {}
        for k in fast:
            g = grads[k]
            m = b1 * state["m"][k] + (1.0 - b1) * g
            v = b2 * state["v"][k] + (1.0 - b2) * g * g
            m_hat = m / bc1
            v_hat = v / bc2
            update = m_hat / torch.sqrt(v_hat + eps * eps)
            new_m[k] = m
            new_v[k] = v
            new_fast[k] = fast[k] - _lr(cfg, k, lr_scale, lr_mult) * update
        return new_fast, {"m": new_m, "v": new_v, "t": t, "bias_correct": state["bias_correct"]}


def newton_schulz5(g: Tensor) -> Tensor:
    """5-step Newton-Schulz orthogonalisation of Keller Jordan's Muon.

        X = G / (||G||_F + 1e-7), transposed if rows > cols
        repeat 5x:  A = X X^T ; B = b*A + c*A@A ; X = a*X + B@X
        with (a, b, c) = (3.4445, -4.7750, 2.0315)

    The quintic is applied in the input dtype (no bf16 downcast as in the reference
    implementation): the outer loop differentiates through it twice, so precision here
    is worth more than speed, and the fast-weight tensors are small.
    """
    assert g.ndim == 2, f"newton_schulz5 expects a 2-D tensor, got shape {tuple(g.shape)}"
    a, b, c = 3.4445, -4.7750, 2.0315

    transposed = g.shape[0] > g.shape[1]
    x = g.mT if transposed else g
    x = x / (x.reshape(-1).norm() + 1e-7)
    for _ in range(5):
        aa = x @ x.mT
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x
    return x.mT if transposed else x


class MuonNoMomentum(InnerOptimizer):
    """W <- W - lr * m_t * sqrt(max(rows, cols)) * NS5(g)

    NS5 is the 5-step Newton-Schulz orthogonalisation of Keller Jordan's Muon (see
    `newton_schulz5`). NS5(g) has per-element RMS 1/sqrt(max(m,n)), so the
    sqrt(max(m,n)) factor again makes `lr` the per-element RMS of the update
    (consistent unit with the other two optimizers).

    Only applies to 2-D tensors; 1-D tensors (norms, biases) have no meaningful
    orthogonalisation and fall back to the NormalizedSGD per-tensor rule
        W <- W - lr * m_t * sqrt(numel(W)) * g / (||g||_F + eps_norm)
    which has the same RMS = lr unit. State is empty.

    The eps_norm floor of NormalizedSGD is applied to the 2-D path as well: NS5
    divides by ||G||_F + 1e-7, so a vanishing chunk gradient would otherwise produce
    an arbitrary orthogonal direction and an enormous meta-gradient.
    """

    def init_state(self, fast: dict[str, Tensor], first_grad: dict[str, Tensor] | None = None) -> dict[str, Any]:
        return {}

    def step(
        self,
        fast: dict[str, Tensor],
        grads: dict[str, Tensor],
        state: dict[str, Any],
        *,
        lr_scale: Tensor | float = 1.0,
        lr_mult: dict[str, Tensor] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Any]]:
        _check_keys(fast, grads, lr_mult)
        cfg = self.cfg

        new_fast: dict[str, Tensor] = {}
        for k, w in fast.items():
            g = grads[k]
            lr = _lr(cfg, k, lr_scale, lr_mult)
            norm = g.reshape(-1).norm()
            gate = (norm >= cfg.eps_norm).to(g.dtype)
            if g.ndim == 2:
                scale = math.sqrt(float(max(g.shape[0], g.shape[1])))
                new_fast[k] = w - lr * scale * gate * newton_schulz5(g)
            else:
                scale = math.sqrt(float(w.numel()))
                new_fast[k] = w - lr * scale * _normalized_direction(g, norm, cfg.eps_norm)
        return new_fast, {}


class ClippedSGD(InnerOptimizer):
    """TTT-E2E's exact inner rule, for the arm E reference.

        W <- W - lr * g / max(1, ||g||_global / tau)

    This is `optax.chain(clip_by_global_norm(tau), sgd(lr))` with tau = 1 and lr = 1,
    which is what every `configs/experiment/*-e2e-*.yaml` in the reference repo sets.
    It differs from NormalizedSGD in exactly one place: when ||g|| < tau the step is
    NOT rescaled, so small late-chunk gradients produce correspondingly small steps
    instead of unit-RMS ones. State is empty.
    """

    def init_state(self, fast: dict[str, Tensor], first_grad: dict[str, Tensor] | None = None) -> dict[str, Any]:
        return {}

    def step(self, fast, grads, state, *, lr_scale=1.0, lr_mult=None):
        _check_keys(fast, grads, lr_mult)
        keys = sorted(fast)
        sq = torch.stack([grads[k].reshape(-1).pow(2).sum() for k in keys]).sum()
        gnorm = sq.sqrt()
        # max(1, ||g||/tau): a differentiable clamp, exactly optax's clip_by_global_norm.
        denom = torch.clamp(gnorm / self.cfg.clip_tau, min=1.0)
        new_fast = {}
        for k in keys:
            new_fast[k] = fast[k] - _lr(self.cfg, k, lr_scale, lr_mult) * grads[k] / denom
        return new_fast, {}


def build_inner_optimizer(cfg: InnerConfig) -> InnerOptimizer:
    """'none' -> a NoOpInnerOptimizer whose step returns fast unchanged."""
    table: dict[str, type[InnerOptimizer]] = {
        "none": NoOpInnerOptimizer,
        "normalized_sgd": NormalizedSGD,
        "adamw": DifferentiableAdamW,
        "muon": MuonNoMomentum,
        "clipped_sgd": ClippedSGD,
    }
    assert cfg.optimizer in table, f"unknown inner optimizer: {cfg.optimizer!r}"
    return table[cfg.optimizer](cfg)
