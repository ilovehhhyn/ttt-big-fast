"""Outer-loop tests."""

from __future__ import annotations

import itertools

import torch

from ttt.config import Config, InnerConfig, LoRAConfig, ModelConfig, OuterConfig, RopeConfig, TrainConfig
from ttt.model.naming import split_parameters
from ttt.model.transformer import TTTTransformer
from ttt.optim.inner import build_inner_optimizer
from ttt.optim.outer import build_outer_optimizer
from ttt.train.inner_loop import TTTInnerLoop
from ttt.train.trainer import Trainer


def build(seqs_per_step=2, lr_rms=1e-2, learned_lr=True, fast_init_trained=False):
    mcfg = ModelConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_layers=3,
                       num_heads=4, num_kv_heads=2, window_size=8, chunk_size=4, fast_blocks=1,
                       rope=RopeConfig(theta=10000.0, scaling="none"), lora=LoRAConfig(rank=2, alpha=4.0))
    cfg = Config(model=mcfg,
                 inner=InnerConfig(optimizer="normalized_sgd", lr_rms=lr_rms, learned_lr=learned_lr),
                 outer=OuterConfig(lr=1e-2, total_steps=10),
                 train=TrainConfig(seq_len=16, tokens_per_step=16 * seqs_per_step, dtype="fp32",
                                   fast_init_trained=fast_init_trained))
    torch.manual_seed(0)
    model = TTTTransformer(mcfg, max_seq_len=16).double()
    split = split_parameters(model, mcfg, cfg.train)
    loop = TTTInnerLoop(model, cfg, build_inner_optimizer(cfg.inner))
    opt = build_outer_optimizer(split.outer, cfg.outer)
    return cfg, model, split, loop, opt


def batches(cfg, n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    for _ in range(n):
        ids = torch.randint(0, cfg.model.vocab_size, (1, cfg.train.seq_len), generator=g)
        tgt = torch.randint(0, cfg.model.vocab_size, (1, cfg.train.seq_len), generator=g)
        yield {"input_ids": ids, "targets": tgt, "loss_mask": torch.ones_like(tgt, dtype=torch.float64)}


def test_fast_param_names_match_split():
    _, model, split, _, _ = build()
    assert sorted(model.fast_param_names()) == sorted(split.fast)


def test_outer_step_updates_slow_and_leaves_fast_and_frozen_alone():
    cfg, model, split, loop, opt = build()
    before_slow = {k: v.detach().clone() for k, v in split.slow.items()}
    before_fast = {k: v.detach().clone() for k, v in split.fast.items()}
    before_frozen = {k: v.detach().clone() for k, v in split.frozen.items()}
    tr = Trainer(cfg, model, split, loop, opt, batches(cfg), device=torch.device("cpu"))
    # Step 1, not 0: the warmup schedule starts at lr exactly 0 (optax's
    # warmup_cosine_decay_schedule with init_value=0, which e2e also uses), so an
    # outer step at step 0 is a genuine no-op and would not test anything.
    m = tr.train_step(1)
    assert m.loss > 0 and m.grad_norm >= 0
    changed = [k for k, v in split.slow.items() if not torch.equal(v, before_slow[k])]
    assert changed, "no slow parameter moved"
    for k, v in split.fast.items():
        assert torch.equal(v, before_fast[k]), f"inner loop wrote back into W0: {k}"
    for k, v in split.frozen.items():
        assert torch.equal(v, before_frozen[k]), f"frozen parameter changed: {k}"


def test_accumulation_matches_single_batch():
    """Accumulating S sequences must equal averaging their losses in one backward."""
    cfg, model, split, loop, opt = build(seqs_per_step=2)
    slow = [v for _, v in sorted(split.slow.items())]

    tr = Trainer(cfg, model, split, loop, opt, batches(cfg), device=torch.device("cpu"))
    opt.zero_grad(set_to_none=True)
    acc = []
    for b in itertools.islice(batches(cfg), 2):
        out = loop.run_sequence(b["input_ids"], b["targets"], b["loss_mask"], dict(split.fast),
                                lr_scale=1.0, lr_mult=tr._lr_mult())
        (out.loss / 2).backward()
    acc = [p.grad.detach().clone() for p in slow]

    opt.zero_grad(set_to_none=True)
    losses = []
    for b in itertools.islice(batches(cfg), 2):
        out = loop.run_sequence(b["input_ids"], b["targets"], b["loss_mask"], dict(split.fast),
                                lr_scale=1.0, lr_mult=tr._lr_mult())
        losses.append(out.loss)
    (torch.stack(losses).mean()).backward()
    single = [p.grad.detach().clone() for p in slow]

    for a, s in zip(acc, single, strict=True):
        assert torch.allclose(a, s, atol=1e-10), (a - s).abs().max()


def test_learned_inner_lr_receives_gradient():
    """The learned per-tensor inner LR is meta-learned: it must get a gradient."""
    cfg, model, split, loop, opt = build(learned_lr=True, lr_rms=1e-1)
    tr = Trainer(cfg, model, split, loop, opt, batches(cfg), device=torch.device("cpu"))
    tr.train_step(1)
    grads = [p.grad for p in model.inner_lr_log.parameters()]
    assert all(g is not None for g in grads), "inner_lr_log got no gradient"
    assert max(g.abs().max().item() for g in grads) > 0, "inner_lr_log gradient is identically zero"


def test_loss_decreases_on_a_memorisable_batch():
    """Sanity: repeatedly meta-training on ONE fixed sequence must reduce its loss."""
    cfg, model, split, loop, opt = build(seqs_per_step=1, lr_rms=1e-2)
    fixed = next(iter(batches(cfg)))
    tr = Trainer(cfg, model, split, loop, opt, itertools.repeat(fixed), device=torch.device("cpu"))
    first = tr.train_step(1).loss
    for s in range(2, 10):
        last = tr.train_step(s).loss
    assert last < first, f"loss did not decrease: {first:.4f} -> {last:.4f}"


def test_arm_d_full_slow_step_trains_the_base_weights():
    """Arm D (slow_spec=("**",), no LoRA): the outer loop owns every non-fast parameter.

    Runs under truncated BPTT with a per-window backward, the path it needs at 32K. The
    fast weights' initialisation W0 is NOT trained: the split is disjoint, the outer
    optimizer owns only the slow set, so W0 stays the pretrained value in this arm.
    """
    mcfg = ModelConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_layers=3,
                       num_heads=4, num_kv_heads=2, window_size=8, chunk_size=4, fast_blocks=1,
                       rope=RopeConfig(theta=10000.0, scaling="none"), lora=None)
    cfg = Config(model=mcfg,
                 inner=InnerConfig(optimizer="normalized_sgd", lr_rms=1e-2, learned_lr=False),
                 outer=OuterConfig(lr=1e-2, total_steps=10),
                 train=TrainConfig(seq_len=16, tokens_per_step=32, dtype="fp32",
                                   truncate_bptt=2, slow_spec=("**",)))
    torch.manual_seed(0)
    model = TTTTransformer(mcfg, max_seq_len=16).double()
    split = split_parameters(model, mcfg, cfg.train)
    assert not split.frozen and split.slow and split.fast
    loop = TTTInnerLoop(model, cfg, build_inner_optimizer(cfg.inner))
    opt = build_outer_optimizer(split.slow, cfg.outer)

    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    Trainer(cfg, model, split, loop, opt, batches(cfg), device=torch.device("cpu")).train_step(1)

    moved = {k for k, v in model.named_parameters() if not torch.equal(v, before[k])}
    assert any("embed_tokens" in k for k in moved), "the embedding did not train"
    assert any(".attn.wq." in k for k in moved), "attention weights did not train"
    assert any(k.startswith("blocks.0.mlp.") for k in moved), "a NON-fast block's MLP did not train"
    assert not (moved & set(split.fast)), f"W0 must not move in this arm: {sorted(moved & set(split.fast))}"


def test_fast_init_trained_moves_w0_and_the_inner_loop_still_starts_from_it():
    """Arm F: the outer loop owns the fast weights' initial value W_0. One outer step moves
    every fast tensor; the frozen set stays put; and the next sequence still starts its
    inner loop from the new W_0 (the live parameter), not from a stale copy."""
    cfg, model, split, loop, opt = build(fast_init_trained=True)
    before_fast = {k: v.detach().clone() for k, v in split.fast.items()}
    before_frozen = {k: v.detach().clone() for k, v in split.frozen.items()}
    assert set(split.outer) == set(split.slow) | set(split.fast)
    tr = Trainer(cfg, model, split, loop, opt, batches(cfg), device=torch.device("cpu"))

    tr.train_step(1)

    for k, v in split.fast.items():
        assert not torch.equal(v, before_fast[k]), f"W_0 did not move: {k}"
    for k, v in split.frozen.items():
        assert torch.equal(v, before_frozen[k]), f"frozen parameter changed: {k}"
    ids, tgt, mask = next(iter(batches(cfg, n=1, seed=5))).values()
    out = loop.run_sequence(ids, tgt, mask, dict(split.fast), inference=True)
    assert torch.isfinite(out.loss)
    for k, v in split.fast.items():
        assert not torch.equal(v, before_fast[k]), "run_sequence must read the live W_0"
