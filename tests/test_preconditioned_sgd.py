"""preconditioned_sgd scales down the shared key directions of a fast-weight update, and nothing else.

Oracle: the update is recomputed from the explicit [in, in] matrix M = I - (1 - c) E E^T as
-lr * sqrt(n) * (G M) / (||G M|| + eps_norm), which shares no code with the optimizer's factored
(G E) E^T form.
All arithmetic is float64; tolerances are rtol = 1e-9, atol = 1e-12 unless a test states otherwise.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from ttt.config import Config, InnerConfig, LoRAConfig, ModelConfig, RopeConfig, TrainConfig
from ttt.model.naming import split_parameters
from ttt.model.transformer import TTTTransformer
from ttt.optim.inner import NormalizedSGD, PreconditionedSGD, build_inner_optimizer
from ttt.optim.key_basis import KEY_BASIS_VERSION, load_key_basis, save_key_basis
from ttt.train.inner_loop import TTTInnerLoop

DT = torch.float64
RTOL, ATOL = 1e-9, 1e-12
LR = 1e-2
OUT, IN, RANK = 6, 8, 3


def orthonormal(rows: int, rank: int, seed: int) -> torch.Tensor:
    """[rows, rank] with orthonormal columns."""
    gen = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(rows, rank, generator=gen, dtype=DT)).Q


def matrices(seed: int) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(OUT, IN, generator=gen, dtype=DT), "b": torch.randn(OUT + 1, IN, generator=gen, dtype=DT)}


def optimizer(shared_keep: float, basis: dict[str, torch.Tensor] | None = None) -> PreconditionedSGD:
    cfg = InnerConfig(optimizer="preconditioned_sgd", lr_rms=LR, key_basis_path="unused: the basis is passed in", shared_keep=shared_keep)
    return PreconditionedSGD(cfg, {"a": orthonormal(IN, RANK, 1), "b": orthonormal(IN, RANK, 2)} if basis is None else basis)


def update(opt: PreconditionedSGD, fast: dict[str, torch.Tensor], grads: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    new_fast, state = opt.step(fast, grads, opt.init_state(fast))
    assert state == {}
    return {k: new_fast[k] - fast[k] for k in fast}


@pytest.mark.parametrize("shared_keep", [0.0, 0.25, 1.0])
def test_update_equals_normalized_gradient_times_the_explicit_key_matrix(shared_keep: float) -> None:
    fast, grads = matrices(0), matrices(1)
    opt = optimizer(shared_keep)

    got = update(opt, fast, grads)

    for k, g in grads.items():
        e = opt.key_basis[k]
        d = g @ (torch.eye(IN, dtype=DT) - (1.0 - shared_keep) * (e @ e.mT))
        want = -LR * math.sqrt(g.numel()) * d / (d.norm() + opt.cfg.eps_norm)
        assert torch.allclose(got[k], want, rtol=RTOL, atol=ATOL), (k, (got[k] - want).abs().max().item())


def test_full_keep_is_normalized_sgd_exactly() -> None:
    fast, grads = matrices(0), matrices(1)
    plain = NormalizedSGD(InnerConfig(optimizer="normalized_sgd", lr_rms=LR))

    got, _ = optimizer(1.0).step(fast, grads, {})
    want, _ = plain.step(fast, grads, {})

    for k in fast:
        assert torch.equal(got[k], want[k]), k


def test_zero_keep_leaves_no_shared_component_and_keeps_the_rms_at_lr() -> None:
    fast, grads = matrices(0), matrices(1)
    opt = optimizer(0.0)

    got = update(opt, fast, grads)

    for k, delta in got.items():
        shared_before = (grads[k] @ opt.key_basis[k]).norm().item()
        assert shared_before > 0.1, f"{k}: the gradient has no shared component, so the test would prove nothing"
        assert (delta @ opt.key_basis[k]).abs().max().item() <= ATOL, k
        assert delta.pow(2).mean().sqrt().item() == pytest.approx(LR, rel=1e-6), k  # eps_norm = 1e-6 sits in the denominator


def test_inputs_are_not_modified() -> None:
    fast, grads = matrices(0), matrices(1)
    fast_before, grads_before = {k: v.clone() for k, v in fast.items()}, {k: v.clone() for k, v in grads.items()}

    optimizer(0.0).step(fast, grads, {})

    for k in fast:
        assert torch.equal(fast[k], fast_before[k]) and torch.equal(grads[k], grads_before[k]), k


def test_meta_gradient_reaches_the_gradient_inputs_and_the_learned_step_size() -> None:
    x = torch.randn(OUT, IN, dtype=DT, generator=torch.Generator().manual_seed(3), requires_grad=True)
    w = torch.randn(OUT, IN, dtype=DT, generator=torch.Generator().manual_seed(4), requires_grad=True)
    log_lr = torch.zeros((), dtype=DT, requires_grad=True)
    (g,) = torch.autograd.grad(0.5 * (w * x).pow(2).sum() + (w * x.sin()).sum(), w, create_graph=True)

    new_fast, _ = optimizer(0.0, {"w": orthonormal(IN, RANK, 5)}).step({"w": w}, {"w": g}, {}, lr_mult={"w": log_lr.exp()})
    d_x, d_log_lr = torch.autograd.grad(new_fast["w"].pow(2).sum(), [x, log_lr])

    for name, grad in (("x", d_x), ("log_lr", d_log_lr)):
        assert torch.isfinite(grad).all() and grad.abs().max().item() > 0.0, name


def test_step_rejects_a_basis_that_names_other_matrices() -> None:
    with pytest.raises(AssertionError, match="must name the same matrices"):
        optimizer(0.0, {"a": orthonormal(IN, RANK, 1)}).step(matrices(0), matrices(1), {})


def test_step_rejects_a_basis_with_the_wrong_input_size() -> None:
    basis = {"a": orthonormal(IN + 1, RANK, 1), "b": orthonormal(IN, RANK, 2)}
    with pytest.raises(AssertionError, match=f"has {IN + 1} input features"):
        optimizer(0.0, basis).step(matrices(0), matrices(1), {})


def test_step_rejects_a_vector_fast_weight() -> None:
    fast = {"a": torch.randn(IN, dtype=DT)}
    with pytest.raises(AssertionError, match="input features but the gradient is"):
        optimizer(0.0, {"a": orthonormal(IN, RANK, 1)}).step(fast, {"a": torch.randn(IN, dtype=DT)}, {})


@pytest.mark.parametrize("kwargs, message", [
    (dict(optimizer="preconditioned_sgd"), "requires a key basis"),
    (dict(optimizer="normalized_sgd", key_basis_path="basis.pt"), "no other optimizer uses one"),
    (dict(optimizer="preconditioned_sgd", key_basis_path="basis.pt", shared_keep=1.5), "shared_keep must be in"),
])
def test_config_rejects_settings_it_cannot_honour(kwargs: dict, message: str) -> None:
    with pytest.raises(AssertionError, match=message):
        InnerConfig(lr_rms=LR, **kwargs)


def test_constructor_rejects_the_global_norm_scope() -> None:
    cfg = InnerConfig(optimizer="preconditioned_sgd", lr_rms=LR, key_basis_path="unused", norm_scope="global")
    with pytest.raises(AssertionError, match="norm_scope must be 'tensor'"):
        PreconditionedSGD(cfg, {"a": orthonormal(IN, RANK, 1)})


def test_a_saved_basis_loads_back_unchanged(tmp_path: Path) -> None:
    basis = {"a": orthonormal(IN, RANK, 1).float(), "b": orthonormal(IN, RANK, 2).float()}

    save_key_basis(tmp_path / "basis.pt", basis, {"tokens": 7})
    loaded = load_key_basis(tmp_path / "basis.pt")

    assert sorted(loaded) == ["a", "b"] and all(torch.equal(loaded[k], basis[k]) for k in basis)
    assert not (tmp_path / "basis.pt.tmp").exists()


def test_save_rejects_columns_that_are_not_orthonormal_and_writes_nothing(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="must have orthonormal columns"):
        save_key_basis(tmp_path / "basis.pt", {"a": 2.0 * orthonormal(IN, RANK, 1)}, {})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("payload, message", [
    ({"version": KEY_BASIS_VERSION + 1, "basis": {"a": orthonormal(IN, RANK, 1)}, "meta": {}}, "this code reads version"),
    ({"version": KEY_BASIS_VERSION, "basis": {"a": orthonormal(IN, RANK, 1)}}, "must hold exactly the fields"),
    ({"version": KEY_BASIS_VERSION, "basis": {"a": orthonormal(IN, RANK, 1)}, "meta": {}, "extra": 1}, "must hold exactly the fields"),
    ({"version": KEY_BASIS_VERSION, "basis": {"a": torch.ones(IN, dtype=DT)}, "meta": {}}, "must be a 2-D floating tensor"),
    ({"version": KEY_BASIS_VERSION, "basis": {}, "meta": {}}, "holds no matrices"),
])
def test_load_rejects_a_malformed_file(tmp_path: Path, payload: dict, message: str) -> None:
    torch.save(payload, tmp_path / "basis.pt")
    with pytest.raises(AssertionError, match=message):
        load_key_basis(tmp_path / "basis.pt")


def test_load_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="does not exist; write it with scripts/key_basis.py"):
        load_key_basis(tmp_path / "absent.pt")


def tiny_model_with_basis(tmp_path: Path, *, learned_lr: bool, truncate_bptt: int = 0):
    """The 3-layer float64 model of tests/test_inner_loop.py, with a random basis for its fast matrices."""
    mcfg = ModelConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_layers=3, num_heads=4, num_kv_heads=2,
                       window_size=8, chunk_size=4, fast_blocks=1, rope=RopeConfig(theta=10000.0, scaling="none"),
                       lora=LoRAConfig(rank=2, alpha=4.0))
    torch.manual_seed(0)
    model = TTTTransformer(mcfg, max_seq_len=16).double()
    train = TrainConfig(seq_len=16, tokens_per_step=16, micro_batch=1, remat_group=1, truncate_bptt=truncate_bptt, dtype="fp32")
    split = split_parameters(model, mcfg, train)
    basis = {name: orthonormal(w.shape[1], 2, seed).float() for seed, (name, w) in enumerate(sorted(split.fast.items()))}
    save_key_basis(tmp_path / "basis.pt", basis, {})
    inner = InnerConfig(optimizer="preconditioned_sgd", lr_rms=LR, learned_lr=learned_lr, key_basis_path=str(tmp_path / "basis.pt"))
    cfg = Config(model=mcfg, inner=inner, train=train)
    gen = torch.Generator().manual_seed(0)
    ids, tgt = (torch.randint(0, 32, (1, 16), generator=gen) for _ in range(2))
    return model, split, TTTInnerLoop(model, cfg, build_inner_optimizer(cfg.inner)), (ids, tgt, torch.ones_like(tgt, dtype=DT))


def test_inference_mode_reports_the_same_numbers_as_the_training_path(tmp_path: Path) -> None:
    _, split, loop, batch = tiny_model_with_basis(tmp_path, learned_lr=False)

    train_path = loop.run_sequence(*batch, dict(split.fast))
    infer_path = loop.run_sequence(*batch, dict(split.fast), inference=True)

    assert torch.allclose(train_path.token_nll, infer_path.token_nll, rtol=0.0, atol=ATOL)
    moved = max((train_path.fast_final[k] - split.fast[k]).abs().max().item() for k in split.fast)
    assert moved > 0.0, "the fast weights never moved, so the comparison would prove nothing"
    for k in train_path.fast_final:
        assert torch.allclose(train_path.fast_final[k], infer_path.fast_final[k], rtol=0.0, atol=ATOL), k


@pytest.mark.parametrize("learned_lr", [False, True])
def test_per_window_backward_gives_the_single_backward_meta_gradient(tmp_path: Path, learned_lr: bool) -> None:
    model, split, loop, batch = tiny_model_with_basis(tmp_path, learned_lr=learned_lr, truncate_bptt=2)
    slow = [v for _, v in sorted(split.slow.items())]
    lr_mult = (lambda: model.inner_lr_multipliers()) if learned_lr else (lambda: None)

    reference = loop.run_sequence(*batch, dict(split.fast), lr_mult=lr_mult())
    wanted = torch.autograd.grad(reference.loss, slow, allow_unused=True)
    windowed = loop.run_sequence(*batch, dict(split.fast), lr_mult=lr_mult(), backward_scale=1.0)

    assert windowed.backward_done
    assert any(g is not None and g.abs().max().item() > 0.0 for g in wanted), "no meta-gradient to compare"
    for p, g in zip(slow, wanted, strict=True):
        have, want = (torch.zeros_like(p) if t is None else t for t in (p.grad, g))
        assert torch.allclose(have, want, rtol=0.0, atol=1e-9), (have - want).abs().max().item()
