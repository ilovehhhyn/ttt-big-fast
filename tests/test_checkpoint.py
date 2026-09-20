"""Checkpoint / resume tests.

The contract: a run that is killed and resumed must be INDISTINGUISHABLE from one that
was never interrupted -- same per-step metrics, same final slow weights. Anything less
means a resumed result is a different experiment wearing the same name.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from ttt.config import Config, InnerConfig, LoRAConfig, ModelConfig, OuterConfig, RopeConfig, TrainConfig
from ttt.model.naming import split_parameters
from ttt.model.transformer import TTTTransformer
from ttt.optim.inner import build_inner_optimizer
from ttt.optim.outer import build_outer_optimizer
from ttt.train.checkpoint import load_checkpoint, save_checkpoint, training_fingerprint
from ttt.train.inner_loop import TTTInnerLoop
from ttt.train.trainer import Trainer

CPU = torch.device("cpu")
FP = {"seq_len": 16, "steps": 10, "outer_lr": 1e-2, "seed": 0}


def build(seqs_per_step=2):
    """Tiny float64 CPU model; mirrors tests/test_trainer.py::build. learned_lr is ON so
    the inner-LR scalars (slow parameters in their own param group) are covered too."""
    mcfg = ModelConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_layers=3,
                       num_heads=4, num_kv_heads=2, window_size=8, chunk_size=4, fast_blocks=1,
                       rope=RopeConfig(theta=10000.0, scaling="none"), lora=LoRAConfig(rank=2, alpha=4.0))
    cfg = Config(model=mcfg,
                 inner=InnerConfig(optimizer="normalized_sgd", lr_rms=1e-2, learned_lr=True),
                 outer=OuterConfig(lr=1e-2, total_steps=10),
                 train=TrainConfig(seq_len=16, tokens_per_step=16 * seqs_per_step, dtype="fp32"))
    torch.manual_seed(0)
    model = TTTTransformer(mcfg, max_seq_len=16).double()
    split = split_parameters(model, mcfg, cfg.train)
    loop = TTTInnerLoop(model, cfg, build_inner_optimizer(cfg.inner))
    opt = build_outer_optimizer(split.slow, cfg.outer)
    return cfg, model, split, loop, opt


def batches(cfg, n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    for _ in range(n):
        ids = torch.randint(0, cfg.model.vocab_size, (1, cfg.train.seq_len), generator=g)
        tgt = torch.randint(0, cfg.model.vocab_size, (1, cfg.train.seq_len), generator=g)
        yield {"input_ids": ids, "targets": tgt, "loss_mask": torch.ones_like(tgt, dtype=torch.float64)}


def _stream(cfg, start: int):
    """Sequence `start`, `start + 1`, ... of one fixed deterministic stream."""
    return itertools.islice(batches(cfg, n=64), start, None)


def _train(trainer: Trainer, steps: range) -> list[dict]:
    logs = []
    for s in steps:
        m = trainer.train_step(s).as_log()
        m.pop("sec_per_step")  # wall-clock: the only field allowed to differ
        logs.append(m)
    return logs


def test_resumed_run_is_indistinguishable_from_an_uninterrupted_one(tmp_path):
    total, cut = 5, 2

    # Reference: `total` steps straight through.
    cfg, model, split, loop, opt = build(seqs_per_step=2)
    ref_logs = _train(Trainer(cfg, model, split, loop, opt, _stream(cfg, 0), device=CPU), range(total))
    ref_slow = {k: v.detach().clone() for k, v in split.slow.items()}

    # Interrupted: `cut` steps, checkpoint, process "dies".
    cfg, model, split, loop, opt = build(seqs_per_step=2)
    logs_a = _train(Trainer(cfg, model, split, loop, opt, _stream(cfg, 0), device=CPU), range(cut))
    ckpt = tmp_path / "run.ckpt"
    save_checkpoint(ckpt, step=cut, split=split, optimizer=opt, history=logs_a, fingerprint=FP)

    # Resumed in a "new process". Scramble the slow weights first: build() is seeded, so
    # without this a load that silently did nothing would still reproduce steps 0..cut
    # and could hide behind an identical initialisation.
    cfg, model, split, loop, opt = build(seqs_per_step=2)
    with torch.no_grad():
        for v in split.slow.values():
            v.add_(torch.randn_like(v))
    step, history = load_checkpoint(ckpt, split=split, optimizer=opt, fingerprint=FP, defaults={})
    assert step == cut and history == logs_a

    start_sequence = step * cfg.train.seqs_per_step  # sequences already consumed
    logs_b = _train(Trainer(cfg, model, split, loop, opt, _stream(cfg, start_sequence), device=CPU),
                    range(step, total))

    assert history + logs_b == ref_logs, "resumed metrics diverge from the uninterrupted run"
    for k, v in split.slow.items():
        assert torch.equal(v, ref_slow[k]), f"resumed slow weight differs: {k}"


def test_optimizer_moments_are_restored_not_reinitialised(tmp_path):
    """AdamW's moments carry the whole optimisation history. Restoring only the weights
    would restart them at zero and quietly change every subsequent step."""
    cfg, model, split, loop, opt = build()
    _train(Trainer(cfg, model, split, loop, opt, _stream(cfg, 0), device=CPU), range(1, 3))
    want = {i: {k: v.clone() if torch.is_tensor(v) else v for k, v in s.items()}
            for i, s in opt.state_dict()["state"].items()}
    assert want, "optimizer has no state to test with"
    ckpt = tmp_path / "run.ckpt"
    save_checkpoint(ckpt, step=3, split=split, optimizer=opt, history=[], fingerprint=FP)

    cfg, model, split, loop, opt = build()
    assert not opt.state_dict()["state"], "fresh optimizer should start empty"
    load_checkpoint(ckpt, split=split, optimizer=opt, fingerprint=FP, defaults={})
    got = opt.state_dict()["state"]
    assert got.keys() == want.keys()
    for i in want:
        for k, v in want[i].items():
            assert torch.equal(got[i][k], v) if torch.is_tensor(v) else got[i][k] == v, (i, k)


def test_resume_refuses_a_different_configuration(tmp_path):
    """Resuming under changed settings would splice two experiments into one result."""
    cfg, model, split, loop, opt = build()
    ckpt = tmp_path / "run.ckpt"
    save_checkpoint(ckpt, step=1, split=split, optimizer=opt, history=[], fingerprint=FP)
    changed = dict(FP, outer_lr=3e-4)
    with pytest.raises(AssertionError, match="outer_lr"):
        load_checkpoint(ckpt, split=split, optimizer=opt, fingerprint=changed, defaults={})


def test_load_refuses_a_checkpoint_for_a_different_parameter_set(tmp_path):
    cfg, model, split, loop, opt = build()
    ckpt = tmp_path / "run.ckpt"
    save_checkpoint(ckpt, step=1, split=split, optimizer=opt, history=[], fingerprint=FP)

    blob = torch.load(ckpt, weights_only=False)
    dropped = sorted(blob["slow"])[0]
    del blob["slow"][dropped]
    torch.save(blob, ckpt)
    with pytest.raises(AssertionError, match="slow parameter names"):
        load_checkpoint(ckpt, split=split, optimizer=opt, fingerprint=FP, defaults={})


def test_a_failed_save_leaves_the_previous_checkpoint_intact(tmp_path, monkeypatch):
    """Jobs die at arbitrary moments, including mid-write. The write must be atomic."""
    cfg, model, split, loop, opt = build()
    ckpt = tmp_path / "run.ckpt"
    save_checkpoint(ckpt, step=1, split=split, optimizer=opt, history=[{"step": 0}], fingerprint=FP)
    good = ckpt.read_bytes()

    def exploding_save(obj, f, *a, **k):
        with open(f, "wb") as fh:
            fh.write(b"partial garbage")
        raise OSError("disk quota exceeded")

    monkeypatch.setattr(torch, "save", exploding_save)
    with pytest.raises(OSError):
        save_checkpoint(ckpt, step=2, split=split, optimizer=opt, history=[], fingerprint=FP)
    monkeypatch.undo()

    assert ckpt.read_bytes() == good, "a failed save corrupted the last good checkpoint"
    step, history = load_checkpoint(ckpt, split=split, optimizer=opt, fingerprint=FP, defaults={})
    assert step == 1 and history == [{"step": 0}]


def test_fingerprint_ignores_operational_args_only():
    a = {"seq_len": 32768, "steps": 20, "out": "a.json", "ckpt": "a.ckpt", "device": "cuda", "hf_cache": None}
    b = dict(a, out="b.json", ckpt="b.ckpt", device="cpu", hf_cache="/x")
    assert training_fingerprint(a) == training_fingerprint(b)
    assert training_fingerprint(a) != training_fingerprint(dict(a, steps=21))
    assert "seq_len" in training_fingerprint(a) and "out" not in training_fingerprint(a)


def test_flag_added_after_the_checkpoint_is_fine_at_its_default_only(tmp_path):
    """Jobs wait days in the queue and the code moves on underneath them. A flag that did
    not exist when the checkpoint was written cannot have influenced that run, so resuming
    is the same experiment exactly when the new flag is still at its default."""
    cfg, model, split, loop, opt = build()
    ckpt = tmp_path / "run.ckpt"
    save_checkpoint(ckpt, step=1, split=split, optimizer=opt, history=[], fingerprint=FP)
    defaults = {"new_flag": 0.0}

    at_default = dict(FP, new_flag=0.0)
    step, _ = load_checkpoint(ckpt, split=split, optimizer=opt, fingerprint=at_default, defaults=defaults)
    assert step == 1

    changed = dict(FP, new_flag=0.5)
    with pytest.raises(AssertionError, match="new_flag"):
        load_checkpoint(ckpt, split=split, optimizer=opt, fingerprint=changed, defaults=defaults)


def test_flag_removed_since_the_checkpoint_is_refused(tmp_path):
    """The reverse is not safe: a setting the old run depended on no longer exists."""
    cfg, model, split, loop, opt = build()
    ckpt = tmp_path / "run.ckpt"
    save_checkpoint(ckpt, step=1, split=split, optimizer=opt, history=[],
                    fingerprint=dict(FP, old_flag=3))
    with pytest.raises(AssertionError, match="old_flag"):
        load_checkpoint(ckpt, split=split, optimizer=opt, fingerprint=FP, defaults={})


def test_load_slow_weights_restores_weights_only(tmp_path):
    """Evaluating trained slow weights under a DIFFERENT inner rule is a legitimate question
    (it fills the 2x2 of 'trained with/without the inner loop' x 'evaluated with/without').
    Unlike a resume it must not demand matching settings, and it must not touch the optimizer."""
    from ttt.train.checkpoint import load_slow_weights

    cfg, model, split, loop, opt = build()
    _train(Trainer(cfg, model, split, loop, opt, _stream(cfg, 0), device=CPU), range(1, 3))
    want = {k: v.detach().clone() for k, v in split.slow.items()}
    ckpt = tmp_path / "run.ckpt"
    save_checkpoint(ckpt, step=3, split=split, optimizer=opt, history=[], fingerprint=FP)

    cfg, model, split, loop, opt = build()
    with torch.no_grad():
        for v in split.slow.values():
            v.add_(torch.randn_like(v))
    info = load_slow_weights(ckpt, split=split)
    for k, v in split.slow.items():
        assert torch.equal(v, want[k]), k
    assert info["step"] == 3 and info["fingerprint"] == FP, "provenance must travel with the weights"
    assert not opt.state_dict()["state"], "loading weights for evaluation must not create optimizer state"


def test_load_slow_weights_refuses_a_different_parameter_set(tmp_path):
    from ttt.train.checkpoint import load_slow_weights

    cfg, model, split, loop, opt = build()
    ckpt = tmp_path / "run.ckpt"
    save_checkpoint(ckpt, step=1, split=split, optimizer=opt, history=[], fingerprint=FP)
    blob = torch.load(ckpt, weights_only=False)
    del blob["slow"][sorted(blob["slow"])[0]]
    torch.save(blob, ckpt)
    with pytest.raises(AssertionError, match="slow parameter names"):
        load_slow_weights(ckpt, split=split)
