"""Outer (meta) optimizer: AdamW, parameter groups, and the two schedules.

The outer loop owns only the SLOW parameters -- LoRA factors, RMSNorm gains and
the per-fast-tensor inner-LR log scalars (see ``ttt.model.naming``). It updates
them once per ``TrainConfig.seqs_per_step`` sequences, because fast weights are
per-sequence state and ``micro_batch`` is pinned to 1: an outer step is
``seqs_per_step`` sequential inner-loop calls with gradient accumulation.

Fixed values (plan 2026-09-17 §0.2, "Outer optimizer" and "Batch size"):

    AdamW betas (0.9, 0.95), weight decay 0.1 on LoRA A/B ONLY,
    grad clip 1.0, 10% linear warmup then cosine to end_lr 1e-5,
    0.5M tokens per outer step (64 x 8192, or 16 x 32768).

Why weight decay is split. LoRA A/B are free parameters with no pretrained
prior, so decay is the usual capacity control. RMSNorm gains start at the
pretrained values and decaying them pulls the model away from its own
initialisation; the inner-LR log scalars start at 0 (multiplier 1) and decaying
them is a prior toward "inner LR = the configured value", which is not a prior
we want to impose silently. Both therefore get wd = 0.

Why the schedule is driven manually. We advance it with ``set_lr`` once per
outer step rather than through ``torch.optim.lr_scheduler``: the outer step is
already a hand-rolled accumulate/clip/step sequence, and a scheduler object adds
a second place where the step counter can drift out of sync with the trainer's.
``lr_at_step`` is a pure function of the step index, so a resumed run recovers
the exact LR from the step number alone.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ttt.config import InnerConfig, OuterConfig

__all__ = [
    "DECAY_PATTERNS",
    "build_outer_optimizer",
    "build_param_groups",
    "inner_lr_scale_at_step",
    "lr_at_step",
    "set_lr",
]

#: Substrings that put a slow parameter in the weight-decayed group. Everything
#: else in the slow set (norm gains, ``inner_lr_log``) gets weight_decay = 0.
DECAY_PATTERNS: tuple[str, ...] = ("lora_A", "lora_B")


def build_param_groups(slow: dict[str, Tensor], cfg: OuterConfig) -> list[dict]:
    """Split the slow parameters into the decayed and undecayed AdamW groups.

    Returns exactly two groups, decayed first:

        [{"params": [...], "names": [...], "weight_decay": cfg.weight_decay},
         {"params": [...], "names": [...], "weight_decay": 0.0}]

    ``names`` is carried for logging and for the partition assertions; PyTorch
    ignores extra keys in a param group.

    Assertions. The two groups must partition ``slow`` exactly (disjoint, total).
    When LoRA is enabled -- detected as "at least one name matches
    ``DECAY_PATTERNS``", since ``OuterConfig`` carries no LoRA fields -- BOTH
    groups must be non-empty: an empty undecayed group means the norm gains and
    inner-LR scalars went missing from the slow set, which would silently decay
    nothing and meta-learn less than intended.
    """
    assert slow, "slow parameter set is empty: nothing for the outer optimizer to own"

    decay_names = sorted(n for n in slow if any(p in n for p in DECAY_PATTERNS))
    other_names = sorted(n for n in slow if not any(p in n for p in DECAY_PATTERNS))

    # Partition check, by name and by tensor identity.
    assert set(decay_names).isdisjoint(other_names), "decay / no-decay groups overlap"
    assert set(decay_names) | set(other_names) == set(slow), "param groups do not cover the slow set"
    ids_decay = {id(slow[n]) for n in decay_names}
    ids_other = {id(slow[n]) for n in other_names}
    assert not (ids_decay & ids_other), "the same tensor appears in both param groups"
    assert len(ids_decay) + len(ids_other) == len(slow), "duplicate tensors in the slow set"

    if decay_names:  # LoRA is enabled
        assert other_names, (
            "LoRA parameters are present but the no-decay group is empty: the slow set "
            "is missing the RMSNorm gains and inner_lr_log scalars "
            f"(got {len(decay_names)} LoRA tensors and nothing else)"
        )

    return [
        {"params": [slow[n] for n in decay_names], "names": decay_names, "weight_decay": cfg.weight_decay},
        {"params": [slow[n] for n in other_names], "names": other_names, "weight_decay": 0.0},
    ]


def build_outer_optimizer(slow: dict[str, Tensor], cfg: OuterConfig) -> torch.optim.AdamW:
    """AdamW over the slow set with the plan's fixed betas and the split weight decay.

    ``lr`` is set to ``cfg.lr`` here only so the groups are well formed; the
    trainer overwrites it every step via ``set_lr(opt, lr_at_step(step, cfg))``.
    """
    groups = build_param_groups(slow, cfg)
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))


def lr_at_step(step: int, cfg: OuterConfig) -> float:
    """Outer learning rate at 0-indexed outer ``step``.

        W = round(cfg.warmup_frac * cfg.total_steps)

        step < W      lr = cfg.lr * step / W                       (linear, 0 at step 0)
        W <= step     p  = (step - W) / (cfg.total_steps - W)
                      lr = end_lr + 0.5 * (cfg.lr - end_lr) * (1 + cos(pi * p))
        step >= T     lr = cfg.end_lr                              (clamped)

    At ``step == W`` the cosine gives ``cos(0) = 1`` and therefore exactly
    ``cfg.lr``; at ``step == cfg.total_steps`` it gives exactly ``cfg.end_lr``.
    The last real training step, ``total_steps - 1``, sits just above ``end_lr``.
    """
    assert step >= 0, f"step must be 0-indexed and non-negative, got {step}"
    total = cfg.total_steps
    warmup = round(cfg.warmup_frac * total)
    assert warmup < total, f"warmup {warmup} must be shorter than total_steps {total}"

    if step < warmup:
        return cfg.lr * step / warmup
    if step >= total:
        return cfg.end_lr
    progress = (step - warmup) / (total - warmup)
    return cfg.end_lr + 0.5 * (cfg.lr - cfg.end_lr) * (1.0 + math.cos(math.pi * progress))


def inner_lr_scale_at_step(step: int, cfg: InnerConfig, total_steps: int) -> float:
    """Inner-LR warmup multiplier at 0-indexed outer ``step`` (e2e's ilr_init / ilr_warmup_steps).

        W = round(cfg.lr_warmup_frac * total_steps)

        W == 0        scale = 1.0
        step < W      scale = 0.1 + 0.9 * step / W      (0.1 at step 0)
        step >= W     scale = 1.0

    A cold inner LR at the start of meta-training keeps the first outer steps
    close to the pretrained model while the LoRA factors are still near zero;
    the fast weights then have something to differentiate through before they
    are allowed to move at full rate.
    """
    assert step >= 0 and total_steps >= 1
    warmup = round(cfg.lr_warmup_frac * total_steps)
    if warmup == 0:
        return 1.0
    if step >= warmup:
        return 1.0
    return 0.1 + 0.9 * step / warmup


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Drive the schedule: overwrite ``lr`` in every param group.

    Both groups share one LR; only the weight decay differs between them.
    """
    assert lr >= 0.0, f"learning rate must be non-negative, got {lr}"
    for group in optimizer.param_groups:
        group["lr"] = lr
