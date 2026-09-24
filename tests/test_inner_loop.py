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


def build(inner: InnerConfig, *, seq_len: int = 16, chunk: int = 4, fast_blocks: int = 1, rank: int = 2,
          token_rates: bool = False):
    mcfg = ModelConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_layers=3,
        num_heads=4, num_kv_heads=2, window_size=8, chunk_size=chunk,
        fast_blocks=fast_blocks, rope=RopeConfig(theta=10000.0, scaling="none"),
        lora=LoRAConfig(rank=rank, alpha=4.0), token_rates=token_rates,
    )
    slow = TrainConfig().slow_spec + (("token_rate",) if token_rates else ())
    cfg = Config(model=mcfg, inner=inner,
                 train=TrainConfig(seq_len=seq_len, tokens_per_step=seq_len, micro_batch=1, dtype="fp32",
                                   slow_spec=slow))
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


def test_resolve_remat_group_defaults_to_one():
    """g=1 by default: measured, g>1 is strictly worse for this loop.

    The sqrt(N) rule assumed a checkpointed group's interior could be discarded. It
    cannot -- the group builds its inner gradient with create_graph=True, and
    torch.utils.checkpoint does not discard a graph created inside the region, so a
    bigger group just holds more second-order graph live (FINDINGS section 13).
    """
    for n in (4, 8, 16, 32, 128, 7):
        assert resolve_remat_group(n, 0) == 1, n
    # An explicit request is still honoured, so the regression stays measurable.
    assert resolve_remat_group(8, 8) == 8
    assert resolve_remat_group(16, 4) == 4


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


def test_truncated_bptt_matches_exact_when_window_covers_everything():
    """truncate_bptt >= num_chunks must be a no-op, and the loss must never change.

    Truncation cuts only the BACKWARD path; the forward values carry through unchanged.
    So the reported loss is identical for every truncation window, while the gradient
    differs once the window is shorter than the sequence.
    """
    cfg, model, split, = build(InnerConfig(optimizer="normalized_sgd", lr_rms=1e-1, learned_lr=False))

    def run(trunc):
        c = Config(model=cfg.model, inner=cfg.inner, outer=cfg.outer,
                   train=TrainConfig(seq_len=cfg.train.seq_len, tokens_per_step=cfg.train.tokens_per_step,
                                     micro_batch=1, remat_group=1, truncate_bptt=trunc, dtype="fp32"))
        loop = TTTInnerLoop(model, c, build_inner_optimizer(c.inner))
        ids, tgt, mask = batch(c)
        out = loop.run_sequence(ids, tgt, mask, dict(split.fast))
        slow = [v for _, v in sorted(split.slow.items())]
        g = torch.autograd.grad(out.loss, slow, allow_unused=True)
        return out.loss, [torch.zeros(1, dtype=torch.float64) if x is None else x for x in g]

    n = cfg.num_chunks
    loss_exact, g_exact = run(0)
    loss_full, g_full = run(n)          # window covers the whole sequence -> identical
    loss_short, g_short = run(1)        # one-chunk window -> genuinely different gradient

    assert torch.allclose(loss_exact, loss_full) and torch.allclose(loss_exact, loss_short), (
        "truncation must not change the forward loss"
    )
    for a, b in zip(g_exact, g_full, strict=True):
        assert torch.allclose(a, b, atol=1e-12), (a - b).abs().max()
    assert max((a - b).abs().max().item() for a, b in zip(g_exact, g_short, strict=True)) > 1e-9, (
        "a one-chunk truncation window must change the meta-gradient"
    )


def test_per_window_backward_matches_single_backward():
    """Per-window TBPTT must accumulate exactly the gradient one backward would.

    Gradients are linear in the loss, so summing the backwards of disjoint window losses
    equals one backward of their sum -- PROVIDED each window's graph reaches the same
    parameters. This is what makes the memory optimisation (freeing a window's graph as
    soon as it is charged) free of any change to the science.
    """
    cfg, model, split = build(InnerConfig(optimizer="normalized_sgd", lr_rms=1e-1, learned_lr=False))
    trunc = 2
    c = Config(model=cfg.model, inner=cfg.inner, outer=cfg.outer,
               train=TrainConfig(seq_len=cfg.train.seq_len, tokens_per_step=cfg.train.tokens_per_step,
                                 micro_batch=1, remat_group=1, truncate_bptt=trunc, dtype="fp32"))
    loop = TTTInnerLoop(model, c, build_inner_optimizer(c.inner))
    ids, tgt, mask = batch(c)
    slow = [v for _, v in sorted(split.slow.items())]

    # Reference: truncation active, but a single backward at the end (backward_scale=None).
    ref = loop.run_sequence(ids, tgt, mask, dict(split.fast))
    assert not ref.backward_done
    g_ref = torch.autograd.grad(ref.loss, slow, allow_unused=True)
    g_ref = [torch.zeros_like(p) if g is None else g for g, p in zip(g_ref, slow, strict=True)]

    # Per-window: run_sequence does the backward itself, into .grad.
    for p in slow:
        p.grad = None
    got = loop.run_sequence(ids, tgt, mask, dict(split.fast), backward_scale=1.0)
    assert got.backward_done, "backward_scale + truncate_bptt must take the per-window path"
    assert not got.loss.requires_grad, "an already-charged loss must not be differentiated again"
    assert torch.allclose(ref.loss, got.loss), "per-window backward must not change the loss"

    for p, g in zip(slow, g_ref, strict=True):
        have = torch.zeros_like(p) if p.grad is None else p.grad
        assert torch.allclose(have, g, atol=1e-10), (have - g).abs().max().item()


@pytest.mark.parametrize("optimizer", ["normalized_sgd", "adamw", "muon", "clipped_sgd"])
def test_inference_mode_reports_identical_numbers(optimizer):
    """inference=True must change only the graph, never a reported value.

    Parametrised over every inner optimizer: AdamW reaches the inner loop through a
    different route (the _first_grad warm start and carried moments), so a fixture that
    only ever ran normalized SGD would leave that route untested.

    It skips create_graph, the checkpointed regions and the prefix graph, all of which
    exist solely to carry the META-gradient. The TTT updates themselves are unchanged,
    so loss, per-chunk loss and token NLL must match the training path exactly.
    """
    cfg, model, split = build(InnerConfig(optimizer=optimizer, lr_rms=1e-2, learned_lr=False))
    loop = TTTInnerLoop(model, cfg, build_inner_optimizer(cfg.inner))
    ids, tgt, mask = batch(cfg)

    train_path = loop.run_sequence(ids, tgt, mask, dict(split.fast))
    infer_path = loop.run_sequence(ids, tgt, mask, dict(split.fast), inference=True)

    assert torch.allclose(train_path.loss, infer_path.loss, atol=1e-12), (
        (train_path.loss - infer_path.loss).abs().item()
    )
    assert torch.allclose(train_path.per_chunk_loss, infer_path.per_chunk_loss, atol=1e-12)
    assert torch.allclose(train_path.token_nll, infer_path.token_nll, atol=1e-12)
    for k in train_path.fast_final:
        assert torch.allclose(train_path.fast_final[k], infer_path.fast_final[k], atol=1e-12), k
    # The point of the mode: no second-order graph is kept.
    assert not infer_path.loss.requires_grad


def test_inference_mode_rejects_backward_scale():
    cfg, model, split = build(InnerConfig(optimizer="normalized_sgd", lr_rms=1e-1, learned_lr=False))
    loop = TTTInnerLoop(model, cfg, build_inner_optimizer(cfg.inner))
    ids, tgt, mask = batch(cfg)
    with pytest.raises(AssertionError, match="nothing for backward_scale to scale"):
        loop.run_sequence(ids, tgt, mask, dict(split.fast), backward_scale=1.0, inference=True)


@pytest.mark.parametrize("optimizer", ["normalized_sgd", "adamw"])
@pytest.mark.parametrize("learned_lr", [False, True])
def test_per_window_backward_matches_single_backward_across_optimizers(optimizer, learned_lr):
    """The prefix cut must hold for stateful inner optimizers too.

    AdamW warm-starts its moments from g_1, which is computed outside the chunk loop and
    travels in the carry. If that gradient is taken w.r.t. prefix_out rather than the
    detached prefix_leaf, the prefix rejoins window 1's graph, window 1's backward frees
    it, and window 2 dies with 'backward through the graph a second time'. So this asserts
    both that it RUNS and that it gives the same gradient as one backward.
    """
    cfg, model, split = build(InnerConfig(optimizer=optimizer, lr_rms=1e-2, learned_lr=learned_lr))
    c = Config(model=cfg.model, inner=cfg.inner, outer=cfg.outer,
               train=TrainConfig(seq_len=cfg.train.seq_len, tokens_per_step=cfg.train.tokens_per_step,
                                 micro_batch=1, remat_group=1, truncate_bptt=2, dtype="fp32"))
    loop = TTTInnerLoop(model, c, build_inner_optimizer(c.inner))
    ids, tgt, mask = batch(c)
    slow = [v for _, v in sorted(split.slow.items())]

    def lr_mult():
        return model.inner_lr_multipliers() if learned_lr else None

    ref = loop.run_sequence(ids, tgt, mask, dict(split.fast), lr_mult=lr_mult())
    g_ref = torch.autograd.grad(ref.loss, slow, allow_unused=True)
    g_ref = [torch.zeros_like(p) if g is None else g for g, p in zip(g_ref, slow, strict=True)]

    for p in slow:
        p.grad = None
    got = loop.run_sequence(ids, tgt, mask, dict(split.fast), lr_mult=lr_mult(), backward_scale=1.0)
    assert got.backward_done
    assert torch.allclose(ref.loss, got.loss), "per-window backward must not change the loss"
    for p, g in zip(slow, g_ref, strict=True):
        have = torch.zeros_like(p) if p.grad is None else p.grad
        assert torch.allclose(have, g, atol=1e-9), (have - g).abs().max().item()


# --------------------------------------------------------------------------- per-token rates


def test_token_rates_at_init_leave_the_sequence_loss_unchanged():
    """eta = 1 at init, so every chunk loss and every inner step is identical."""
    inner = InnerConfig(optimizer="normalized_sgd", lr_rms=1e-2, learned_lr=True)
    cfg_plain, model_plain, split_plain = build(inner)
    cfg_rated, model_rated, split_rated = build(inner, token_rates=True)
    model_rated.load_state_dict(model_plain.state_dict(), strict=False)

    out_plain, _ = meta_grad(cfg_plain, model_plain, split_plain, remat_group=1)
    out_rated, _ = meta_grad(cfg_rated, model_rated, split_rated, remat_group=1)

    torch.testing.assert_close(out_rated.per_chunk_loss, out_plain.per_chunk_loss, rtol=0.0, atol=0.0)
    assert sorted(k for k in split_rated.slow if "token_rate" in k) == [
        "blocks.2.token_rate.linear.bias", "blocks.2.token_rate.linear.weight"]
    assert not any("token_rate" in k for k in split_plain.slow)


def test_token_rate_parameters_receive_a_meta_gradient_through_the_inner_loop():
    """The rates enter only through the inner gradient, so a nonzero meta-gradient on
    them proves the second-order path through scale_gradient is intact."""
    inner = InnerConfig(optimizer="normalized_sgd", lr_rms=1e-1, learned_lr=True)
    cfg, model, split = build(inner, token_rates=True)

    _, grads = meta_grad(cfg, model, split, remat_group=1)

    names = sorted(split.slow)
    g_weight = grads[names.index("blocks.2.token_rate.linear.weight")]
    g_bias = grads[names.index("blocks.2.token_rate.linear.bias")]
    assert torch.isfinite(g_weight).all() and g_weight.abs().max().item() > 0.0, g_weight
    assert torch.isfinite(g_bias).all() and g_bias.abs().max().item() > 0.0, g_bias
