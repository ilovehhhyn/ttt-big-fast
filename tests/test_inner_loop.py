"""End-to-end tests for the TTT inner loop.

The load-bearing invariants:
  1. lr = 0  =>  TTT is a no-op, so the outer gradient equals the plain gradient.
  2. Checkpointing through time changes memory, never numerics: the meta-gradient
     must be identical for every remat_group.
  3. The loss for chunk i is computed with W_{i-1}, never W_i.
  4. Fast weights actually move, and the meta-gradient reaches the slow params
     through the inner gradients (not only through the direct path).
"""

from __future__ import annotations

import pytest
import torch

from ttt.config import Config, InnerConfig, LoRAConfig, ModelConfig, RopeConfig, TrainConfig
from ttt.model.naming import split_parameters
from ttt.model.transformer import TTTTransformer
from ttt.optim.inner import build_inner_optimizer
from ttt.train.inner_loop import TTTInnerLoop, resolve_remat_group


def build(inner: InnerConfig, *, seq_len: int = 16, chunk: int = 4, fast_blocks: int = 1, rank: int = 2):
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


def meta_grad(cfg, model, split, *, remat_group):
    cfg = Config(model=cfg.model, inner=cfg.inner, outer=cfg.outer,
                 train=TrainConfig(seq_len=cfg.train.seq_len, tokens_per_step=cfg.train.tokens_per_step,
                                   micro_batch=1, remat_group=remat_group, dtype="fp32"))
    loop = TTTInnerLoop(model, cfg, build_inner_optimizer(cfg.inner))
    ids, tgt, mask = batch(cfg)
    out = loop.run_sequence(ids, tgt, mask, dict(split.fast))
    slow = [v for _, v in sorted(split.slow.items())]
    grads = torch.autograd.grad(out.loss, slow, allow_unused=True)
    return out, [torch.zeros(1, dtype=torch.float64) if g is None else g for g in grads]


def test_resolve_remat_group_picks_nearest_divisor():
    # Perfect squares: exactly sqrt(N).
    assert resolve_remat_group(16, 0) == 4
    assert resolve_remat_group(4, 0) == 2
    # N=8: sqrt is 2.83 but 3 does not divide 8, so the nearest divisor is 2.
    assert resolve_remat_group(8, 0) == 2
    # N=32: sqrt is 5.66; divisors are 4 and 8, and 4 is nearer.
    assert resolve_remat_group(32, 0) == 4
    # N=128 (128K context): sqrt is 11.3; nearest divisor is 8.
    assert resolve_remat_group(128, 0) == 8
    assert resolve_remat_group(7, 0) == 1  # prime N has only 1 and N
    assert resolve_remat_group(8, 8) == 8


def test_resolve_remat_group_rejects_non_divisor():
    import pytest as _pytest
    with _pytest.raises(AssertionError, match="must divide"):
        resolve_remat_group(8, 3)


def test_zero_inner_lr_matches_plain_gradient():
    """lr=0 => W_i == W_0 for all i, so the meta-gradient must equal the gradient of
    the ordinary (no-TTT) chunked forward. This is the sharpest end-to-end check that
    the inner loop adds nothing spurious."""
    cfg, model, split = build(InnerConfig(optimizer="normalized_sgd", lr_rms=0.0, learned_lr=False))
    _, ttt = meta_grad(cfg, model, split, remat_group=1)

    # Reference: same chunked forward, no inner updates at all.
    loop = TTTInnerLoop(model, cfg, build_inner_optimizer(InnerConfig(optimizer="none")))
    ids, tgt, mask = batch(cfg)
    ref_out = loop.run_sequence(ids, tgt, mask, dict(split.fast))
    slow = [v for _, v in sorted(split.slow.items())]
    ref = [g if g is not None else torch.zeros(1, dtype=torch.float64)
           for g in torch.autograd.grad(ref_out.loss, slow, allow_unused=True)]
    for a, b in zip(ttt, ref, strict=True):
        assert torch.allclose(a, b, atol=1e-12), (a - b).abs().max()


@pytest.mark.parametrize("group", [1, 2, 4])
def test_remat_group_does_not_change_numerics(group):
    """Memory knob must be numerically inert, including through the second-order path."""
    cfg, model, split = build(InnerConfig(optimizer="normalized_sgd", lr_rms=1e-2, learned_lr=False))
    ref_out, ref = meta_grad(cfg, model, split, remat_group=4)
    got_out, got = meta_grad(cfg, model, split, remat_group=group)
    assert torch.allclose(ref_out.loss, got_out.loss, atol=1e-12)
    for a, b in zip(got, ref, strict=True):
        assert torch.allclose(a, b, atol=1e-10), (a - b).abs().max()


def test_fast_weights_move_and_loss_is_before_update():
    cfg, model, split = build(InnerConfig(optimizer="normalized_sgd", lr_rms=1e-2, learned_lr=False))
    out, _ = meta_grad(cfg, model, split, remat_group=1)
    assert out.per_chunk_loss.numel() == cfg.num_chunks
    moved = max((out.fast_final[k] - split.fast[k]).abs().max().item() for k in split.fast)
    assert moved > 0, "fast weights did not move"
    # Chunk 0's loss is evaluated at W_0, so it must equal the no-TTT chunk-0 loss.
    loop = TTTInnerLoop(model, cfg, build_inner_optimizer(InnerConfig(optimizer="none")))
    ids, tgt, mask = batch(cfg)
    ref = loop.run_sequence(ids, tgt, mask, dict(split.fast))
    assert torch.allclose(out.per_chunk_loss[0], ref.per_chunk_loss[0], atol=1e-12)
    assert not torch.allclose(out.per_chunk_loss[-1], ref.per_chunk_loss[-1], atol=1e-9), (
        "later chunks must differ once the fast weights have been updated"
    )


def test_metagradient_uses_second_order_path():
    """Turning the inner LR on must change the slow-parameter gradient. If it does not,
    the second-order term dW/dtheta is being dropped."""
    cfg0, model, split = build(InnerConfig(optimizer="normalized_sgd", lr_rms=0.0, learned_lr=False))
    _, g0 = meta_grad(cfg0, model, split, remat_group=1)
    cfg1 = Config(model=cfg0.model, inner=InnerConfig(optimizer="normalized_sgd", lr_rms=1e-1, learned_lr=False),
                  train=cfg0.train)
    _, g1 = meta_grad(cfg1, model, split, remat_group=1)
    assert max((a - b).abs().max().item() for a, b in zip(g0, g1, strict=True)) > 1e-8


def test_adamw_inner_runs_and_flows():
    cfg, model, split = build(InnerConfig(optimizer="adamw", lr_rms=1e-3, eps=1e-8, warm_start=True, learned_lr=False))
    out, grads = meta_grad(cfg, model, split, remat_group=2)
    assert torch.isfinite(out.loss)
    assert all(torch.isfinite(g).all() for g in grads)
    assert max(g.abs().max().item() for g in grads) > 0


def test_gradient_flows_through_the_kv_cache_across_chunks():
    """The KV cache is part of the differentiable carry, not a detached buffer.

    Keys and values cached during chunk j are computed from hidden states that depend on
    the fast weights of earlier chunks (for every fast block above the first). If the
    cache were detached at chunk boundaries, later chunks' losses would lose their
    dependence on earlier inner updates and the meta-gradient would be silently wrong.

    We detect this by comparing the meta-gradient against a run where the cache IS
    detached between chunks: the two must differ.
    """
    import ttt.model.transformer as tmod

    cfg, model, split = build(InnerConfig(optimizer="normalized_sgd", lr_rms=1e-1, learned_lr=False))
    _, ref = meta_grad(cfg, model, split, remat_group=1)

    # Save the DESCRIPTOR from the class __dict__, not the plain function that attribute
    # access returns: reassigning the bare function would drop the staticmethod wrapper
    # and make every later call pass `self` as the first argument.
    original_desc = tmod.TTTTransformer.__dict__["flatten_caches"]
    original_fn = original_desc.__func__

    def detaching_flatten(caches):
        return tuple(t.detach() for t in original_fn(caches))

    tmod.TTTTransformer.flatten_caches = staticmethod(detaching_flatten)
    try:
        _, detached = meta_grad(cfg, model, split, remat_group=1)
    finally:
        tmod.TTTTransformer.flatten_caches = original_desc

    diff = max((a - b).abs().max().item() for a, b in zip(ref, detached, strict=True))
    assert diff > 1e-9, (
        "detaching the KV cache did not change the meta-gradient, so gradients are not "
        "flowing through the cache across chunk boundaries"
    )
