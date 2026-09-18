"""Evaluation and the forgetting probe.

Two things are easy to get catastrophically wrong here and both are tested:

  1. Wrapping the TTT inner loop in ``torch.no_grad()``. The inner loop calls
     ``torch.autograd.grad`` on the chunk loss, so under no_grad it raises (or,
     worse, silently degenerates). ``evaluate`` must re-enable grad around the
     loop and detach the outputs instead.
  2. Scoring the forgetting probe with a stale or accidentally-updated weight
     set. With ``lr_rms = 0`` we have W_T == W_0 exactly, so the probe delta must
     be exactly 0.0 -- any bookkeeping slip shows up immediately.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ttt.config import Config, InnerConfig, LoRAConfig, ModelConfig, RopeConfig, TrainConfig
from ttt.eval.evaluator import EvalResult, evaluate
from ttt.eval.forgetting import forgetting_delta_nll
from ttt.model.naming import split_parameters
from ttt.model.transformer import TTTTransformer
from ttt.optim.inner import build_inner_optimizer
from ttt.train.inner_loop import TTTInnerLoop


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
    loop = TTTInnerLoop(model, cfg, build_inner_optimizer(inner))
    return cfg, model, split, loop


def make_batch(cfg, *, seed=0, n=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, cfg.model.vocab_size, (n, cfg.train.seq_len), generator=g)
    tgt = torch.randint(0, cfg.model.vocab_size, (n, cfg.train.seq_len), generator=g)
    return {"input_ids": ids, "targets": tgt, "loss_mask": torch.ones_like(tgt, dtype=torch.float64)}


def loader(cfg, *, seeds=(0, 1), n=1):
    """A DataLoader is just an iterable of batch dicts as far as `evaluate` cares."""
    return [make_batch(cfg, seed=s, n=n) for s in seeds]


# ------------------------------------------------------------------- evaluate


def test_evaluate_shapes_and_determinism():
    cfg, _, split, loop = build(InnerConfig(optimizer="normalized_sgd", lr_rms=1e-2, learned_lr=False))
    data = loader(cfg, seeds=(0, 1, 2))

    first = evaluate(loop, split, data)
    second = evaluate(loop, split, data)

    assert isinstance(first, EvalResult)
    assert first.num_sequences == 3
    assert isinstance(first.token_nll, np.ndarray)
    assert first.token_nll.shape == (cfg.train.seq_len,)
    assert np.isfinite(first.token_nll).all()
    assert first.forgetting_delta_nll is None

    assert first.loss == second.loss
    assert np.array_equal(first.token_nll, second.token_nll)
    assert first.loss > 0.0


def test_evaluate_respects_max_sequences_and_batched_loader():
    cfg, _, split, loop = build(InnerConfig(optimizer="normalized_sgd", lr_rms=1e-2, learned_lr=False))
    data = loader(cfg, seeds=(0, 1), n=2)  # 2 batches x 2 sequences
    assert evaluate(loop, split, data).num_sequences == 4
    assert evaluate(loop, split, data, max_sequences=3).num_sequences == 3


def test_evaluate_does_not_disable_grad_for_the_inner_loop():
    """The whole point: TTT needs autograd. If `evaluate` ran under no_grad the
    inner loop's torch.autograd.grad call would raise, so a finite loss here IS
    the assertion. We additionally check no graph survives into the result."""
    cfg, _, split, loop = build(InnerConfig(optimizer="normalized_sgd", lr_rms=1e-2, learned_lr=False))
    res = evaluate(loop, split, loader(cfg, seeds=(0,)))
    assert np.isfinite(res.loss)
    assert isinstance(res.loss, float)


def test_evaluate_resets_fast_weights_per_sequence():
    """Order must not matter: each sequence starts from W_0, so the mean loss over
    {a, b} equals the mean over {b, a} and the model's W_0 is untouched."""
    cfg, model, split, loop = build(InnerConfig(optimizer="normalized_sgd", lr_rms=1e-2, learned_lr=False))
    w0 = {n: p.detach().clone() for n, p in split.fast.items()}
    fwd = evaluate(loop, split, loader(cfg, seeds=(0, 1)))
    rev = evaluate(loop, split, loader(cfg, seeds=(1, 0)))
    assert fwd.loss == pytest.approx(rev.loss, rel=1e-12)
    for n, p in split.fast.items():
        assert torch.equal(p.detach(), w0[n]), f"inner loop wrote back into W0 for {n}"


def test_evaluate_with_probe_reports_forgetting():
    cfg, _, split, loop = build(InnerConfig(optimizer="normalized_sgd", lr_rms=1e-2, learned_lr=False))
    probe = make_batch(cfg, seed=99)
    res = evaluate(loop, split, loader(cfg, seeds=(0, 1)), probe_batch=probe)
    assert res.forgetting_delta_nll is not None
    assert np.isfinite(res.forgetting_delta_nll)


# ------------------------------------------------------------------ forgetting


def test_forgetting_is_zero_when_inner_lr_is_zero():
    """lr_rms = 0 => W_T == W_0 bitwise, so the probe must score identically under
    both and the delta must be exactly zero. This is the sharpest check that the
    probe is really scoring the post-TTT weights and not, say, W_0 twice with a
    different mask, or W_T twice."""
    cfg, _, split, loop = build(InnerConfig(optimizer="normalized_sgd", lr_rms=0.0, learned_lr=False))
    seq = make_batch(cfg, seed=0)
    probe = make_batch(cfg, seed=7)
    delta = forgetting_delta_nll(loop, split, seq, probe)
    assert isinstance(delta, float)
    assert abs(delta) < 1e-12, f"expected exactly 0.0, got {delta}"


def test_forgetting_positive_direction():
    """Large inner LR on random data: the delta must be finite and non-degenerate.
    The SIGN is the experimental question (does TTT forget?), not an invariant, so
    it is deliberately not asserted."""
    cfg, _, split, loop = build(InnerConfig(optimizer="normalized_sgd", lr_rms=3e-1, learned_lr=False))
    seq = make_batch(cfg, seed=0)
    probe = make_batch(cfg, seed=7)
    delta = forgetting_delta_nll(loop, split, seq, probe)
    assert np.isfinite(delta)
    assert abs(delta) > 0.0, "a large inner LR must change the probe score at all"


def test_forgetting_probe_takes_no_inner_step():
    """The probe is scored with a plain chunked forward. If it took an inner step,
    running it twice with the same weights would drift."""
    cfg, _, split, loop = build(InnerConfig(optimizer="normalized_sgd", lr_rms=1e-1, learned_lr=False))
    seq = make_batch(cfg, seed=0)
    probe = make_batch(cfg, seed=7)
    a = forgetting_delta_nll(loop, split, seq, probe)
    b = forgetting_delta_nll(loop, split, seq, probe)
    assert a == b


def test_inner_loop_really_breaks_under_no_grad():
    """The premise behind `evaluate`'s enable_grad block, pinned so nobody
    "simplifies" it away: TTT under torch.no_grad() raises, because the chunk loss
    has no grad_fn for torch.autograd.grad to differentiate."""
    cfg, _, split, loop = build(InnerConfig(optimizer="normalized_sgd", lr_rms=1e-2, learned_lr=False))
    b = make_batch(cfg, seed=0)
    with torch.no_grad(), pytest.raises(RuntimeError, match="does not require grad"):
        loop.run_sequence(b["input_ids"], b["targets"], b["loss_mask"], dict(split.fast))
