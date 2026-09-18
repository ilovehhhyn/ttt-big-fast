"""Outer-loop optimizer: schedules, parameter groups, and the accumulation contract.

The invariants that matter:
  1. The LR schedule is exactly 10% linear warmup then cosine to end_lr -- no
     off-by-one at the warmup boundary, and clamped past the end.
  2. Weight decay reaches LoRA A/B and NOTHING else (norm gains and the inner-LR
     log scalars must stay at wd=0, per plan 0.2).
  3. An outer step moves every slow tensor and no other tensor. In particular the
     inner loop must not write back into W_0: the fast parameters of the MODEL are
     the reset point for every sequence, and the outer optimizer does not own them.
  4. Gradient accumulation over `seqs_per_step` sequences is numerically the same
     as one large batch. micro_batch is 1 by construction, so this is the only
     thing standing between us and the paper's 0.5M-token batch.
"""

from __future__ import annotations

import math

import pytest
import torch

from ttt.config import Config, InnerConfig, LoRAConfig, ModelConfig, OuterConfig, RopeConfig, TrainConfig
from ttt.model.naming import split_parameters
from ttt.model.transformer import TTTTransformer
from ttt.optim.inner import build_inner_optimizer
from ttt.optim.outer import (
    build_outer_optimizer,
    build_param_groups,
    inner_lr_scale_at_step,
    lr_at_step,
    set_lr,
)
from ttt.train.inner_loop import TTTInnerLoop


# --------------------------------------------------------------------- fixtures


def build(inner: InnerConfig, *, seq_len: int = 16, chunk: int = 4, fast_blocks: int = 1, rank: int = 2):
    """Tiny float64 CPU model; mirrors tests/test_inner_loop.py::build."""
    mcfg = ModelConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_layers=3,
        num_heads=4, num_kv_heads=2, window_size=8, chunk_size=chunk,
        fast_blocks=fast_blocks, rope=RopeConfig(theta=10000.0, scaling="none"),
        lora=LoRAConfig(rank=rank, alpha=4.0),
    )
    cfg = Config(model=mcfg, inner=inner,
                 train=TrainConfig(seq_len=seq_len, tokens_per_step=seq_len, micro_batch=1, dtype="fp32"))
    torch.manual_seed(0)
    model = TTTTransformer(mcfg, max_seq_len=seq_len).double()
    split = split_parameters(model, mcfg, cfg.train)
    return cfg, model, split


def batch(cfg, *, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, cfg.model.vocab_size, (1, cfg.train.seq_len), generator=g)
    tgt = torch.randint(0, cfg.model.vocab_size, (1, cfg.train.seq_len), generator=g)
    return ids, tgt, torch.ones_like(tgt, dtype=torch.float64)


# ------------------------------------------------------------------ schedules


def test_lr_schedule_shape():
    cfg = OuterConfig(lr=1e-3, end_lr=1e-5, warmup_frac=0.1, total_steps=100)
    warmup = round(cfg.warmup_frac * cfg.total_steps)
    assert warmup == 10

    # step 0 is the first warmup point: the schedule starts at zero and ramps.
    assert lr_at_step(0, cfg) == 0.0
    for s in range(1, warmup):
        expected = cfg.lr * s / warmup
        assert lr_at_step(s, cfg) == pytest.approx(expected, rel=1e-12)
    # exact peak at the end of warmup -- no off-by-one, no overshoot
    assert lr_at_step(warmup, cfg) == pytest.approx(cfg.lr, rel=1e-12)

    # cosine leg: strictly decreasing down to ~end_lr
    tail = [lr_at_step(s, cfg) for s in range(warmup, cfg.total_steps)]
    assert all(a > b for a, b in zip(tail, tail[1:])), "cosine leg must decrease monotonically"
    assert tail[-1] == pytest.approx(cfg.end_lr, abs=1e-6)
    assert lr_at_step(cfg.total_steps, cfg) == pytest.approx(cfg.end_lr, rel=1e-12)

    # clamped past the end
    for s in (cfg.total_steps, cfg.total_steps + 1, 10 * cfg.total_steps):
        assert lr_at_step(s, cfg) == pytest.approx(cfg.end_lr, rel=1e-12)

    # never leaves [end_lr, lr] anywhere on the cosine leg
    assert all(cfg.end_lr - 1e-15 <= v <= cfg.lr + 1e-15 for v in tail)


def test_lr_schedule_no_warmup():
    cfg = OuterConfig(lr=1e-3, end_lr=1e-5, warmup_frac=0.0, total_steps=50)
    assert lr_at_step(0, cfg) == pytest.approx(cfg.lr, rel=1e-12)
    assert lr_at_step(50, cfg) == pytest.approx(cfg.end_lr, rel=1e-12)


def test_inner_lr_scale():
    total = 100
    inner = InnerConfig(lr_warmup_frac=0.1)
    warmup = round(inner.lr_warmup_frac * total)
    assert warmup == 10

    assert inner_lr_scale_at_step(0, inner, total) == pytest.approx(0.1, rel=1e-12)
    assert inner_lr_scale_at_step(warmup, inner, total) == pytest.approx(1.0, rel=1e-12)
    for s in (warmup, warmup + 1, total, 10 * total):
        assert inner_lr_scale_at_step(s, inner, total) == pytest.approx(1.0, rel=1e-12)

    ramp = [inner_lr_scale_at_step(s, inner, total) for s in range(warmup + 1)]
    assert all(a < b for a, b in zip(ramp, ramp[1:])), "ramp must be strictly increasing"
    assert ramp[5] == pytest.approx(0.1 + 0.9 * 0.5, rel=1e-12)

    off = InnerConfig(lr_warmup_frac=0.0)
    assert all(inner_lr_scale_at_step(s, off, total) == 1.0 for s in (0, 1, 7, total, 10 * total))


# ---------------------------------------------------------------- param groups


def test_param_groups_partition():
    slow = {
        "blocks.0.attn.wq.lora_A": torch.zeros(2, 4, requires_grad=True),
        "blocks.0.attn.wq.lora_B": torch.zeros(4, 2, requires_grad=True),
        "blocks.1.attn.wo.lora_A": torch.zeros(2, 4, requires_grad=True),
        "blocks.0.seq_norm.weight": torch.ones(4, requires_grad=True),
        "norm.weight": torch.ones(4, requires_grad=True),
        "blocks.2.inner_lr_log": torch.zeros((), requires_grad=True),
    }
    cfg = OuterConfig(weight_decay=0.1)
    groups = build_param_groups(slow, cfg)
    assert len(groups) == 2

    decay = next(g for g in groups if g["weight_decay"] == cfg.weight_decay)
    nodecay = next(g for g in groups if g["weight_decay"] == 0.0)
    assert set(decay["names"]) == {
        "blocks.0.attn.wq.lora_A", "blocks.0.attn.wq.lora_B", "blocks.1.attn.wo.lora_A",
    }
    assert set(nodecay["names"]) == {
        "blocks.0.seq_norm.weight", "norm.weight", "blocks.2.inner_lr_log",
    }

    # union == input, no overlap, and the LIVE tensors (identity, not equality)
    ids_decay = {id(p) for p in decay["params"]}
    ids_nodecay = {id(p) for p in nodecay["params"]}
    assert not (ids_decay & ids_nodecay)
    assert ids_decay | ids_nodecay == {id(p) for p in slow.values()}

    opt = build_outer_optimizer(slow, cfg)
    assert isinstance(opt, torch.optim.AdamW)
    assert [g["weight_decay"] for g in opt.param_groups] == [cfg.weight_decay, 0.0]
    assert all(g["betas"] == (cfg.beta1, cfg.beta2) for g in opt.param_groups)
    assert all(g["lr"] == cfg.lr for g in opt.param_groups)

    set_lr(opt, 1.234e-4)
    assert all(g["lr"] == 1.234e-4 for g in opt.param_groups)


def test_param_groups_reject_empty_side():
    only_lora = {"a.lora_A": torch.zeros(2, requires_grad=True)}
    with pytest.raises(AssertionError):
        build_param_groups(only_lora, OuterConfig())


# ------------------------------------------------------------------ outer step


def test_outer_step_changes_only_slow():
    cfg, model, split = build(InnerConfig(optimizer="normalized_sgd", lr_rms=1e-2, learned_lr=True))
    loop = TTTInnerLoop(model, cfg, build_inner_optimizer(cfg.inner))
    ids, tgt, mask = batch(cfg)

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = build_outer_optimizer(split.slow, cfg.outer)

    out = loop.run_sequence(ids, tgt, mask, dict(split.fast),
                            lr_mult=model.inner_lr_multipliers())
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(list(split.slow.values()), cfg.outer.grad_clip)
    set_lr(opt, lr_at_step(0, OuterConfig(lr=1e-2, warmup_frac=0.0, total_steps=100)))
    opt.step()

    for name in split.slow:
        moved = (model.get_parameter(name) - before[name]).abs().max().item()
        assert moved > 0.0, f"slow parameter {name} did not move"
    for name in split.frozen:
        assert torch.equal(model.get_parameter(name), before[name]), f"frozen {name} moved"
    for name in split.fast:
        # W_0 is the per-sequence reset point; the inner loop is functional and must
        # never write back into it, and the outer optimizer does not own it either.
        assert torch.equal(model.get_parameter(name), before[name]), f"fast W0 {name} moved"


def test_gradient_accumulation_equals_large_batch():
    """0.5M tokens/step = 64 sequential inner-loop calls. If accumulation did not
    reproduce the large-batch gradient exactly, the batch-size decision would be a lie."""
    cfg, model, split = build(InnerConfig(optimizer="normalized_sgd", lr_rms=1e-2, learned_lr=True))
    loop = TTTInnerLoop(model, cfg, build_inner_optimizer(cfg.inner))
    b0 = batch(cfg, seed=0)
    b1 = batch(cfg, seed=1)
    names = sorted(split.slow)

    def zero():
        for p in split.slow.values():
            p.grad = None

    # one-shot: (l0 + l1) / 2
    zero()
    l0 = loop.run_sequence(*b0, dict(split.fast), lr_mult=model.inner_lr_multipliers()).loss
    l1 = loop.run_sequence(*b1, dict(split.fast), lr_mult=model.inner_lr_multipliers()).loss
    ((l0 + l1) / 2).backward()
    single = {n: split.slow[n].grad.detach().clone() for n in names}

    # accumulated: l0/2 then l1/2, two independent backward passes
    zero()
    (loop.run_sequence(*b0, dict(split.fast), lr_mult=model.inner_lr_multipliers()).loss / 2).backward()
    (loop.run_sequence(*b1, dict(split.fast), lr_mult=model.inner_lr_multipliers()).loss / 2).backward()
    accum = {n: split.slow[n].grad.detach().clone() for n in names}

    for n in names:
        assert single[n].dtype == torch.float64
        diff = (single[n] - accum[n]).abs().max().item()
        assert diff < 1e-10, f"{n}: accumulation differs by {diff}"
    assert max(single[n].abs().max().item() for n in names) > 0.0, "all gradients were zero"


def test_grad_clip_matches_config():
    """Sanity: the clip value we drive the outer step with is the plan's 1.0."""
    assert OuterConfig().grad_clip == 1.0
    assert OuterConfig().beta1 == 0.9 and OuterConfig().beta2 == 0.95
    assert math.isclose(OuterConfig().weight_decay, 0.1)
