"""Tests for ttt.model.naming (Task 3b): the fast / slow / frozen partition."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from ttt.config import LoRAConfig, ModelConfig, TrainConfig
from ttt.model.naming import ParamSplit, is_fast_param, is_slow_param, split_parameters


# --------------------------------------------------------------------------
# A dummy model that reproduces the real naming convention without importing
# the (separately owned) transformer implementation.
# --------------------------------------------------------------------------
class _FakeLoRALinear(nn.Module):
    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim))
        self.lora_A = nn.Parameter(torch.randn(rank, dim))
        self.lora_B = nn.Parameter(torch.zeros(dim, rank))


class _FakeLinear(nn.Module):
    def __init__(self, out_dim: int, in_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))


class _FakeAttn(nn.Module):
    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.wq = _FakeLoRALinear(dim, rank)
        self.wk = _FakeLoRALinear(dim, rank)
        self.wo = _FakeLoRALinear(dim, rank)


class _FakeMLP(nn.Module):
    def __init__(self, dim: int, hidden: int, lora_rank: int = 0) -> None:
        super().__init__()
        if lora_rank:
            self.w1 = _FakeLoRALinear(dim, lora_rank)
            self.w2 = _FakeLoRALinear(dim, lora_rank)
            self.w3 = _FakeLoRALinear(dim, lora_rank)
        else:
            self.w1 = _FakeLinear(hidden, dim)
            self.w2 = _FakeLinear(dim, hidden)
            self.w3 = _FakeLinear(hidden, dim)


class _FakeBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, rank: int, mlp_lora_rank: int = 0) -> None:
        super().__init__()
        self.seq_norm = nn.RMSNorm(dim)
        self.mlp_norm = nn.RMSNorm(dim)
        self.attn = _FakeAttn(dim, rank)
        self.mlp = _FakeMLP(dim, hidden, mlp_lora_rank)
        self.inner_lr_log = nn.Parameter(torch.zeros(()))


class _FakeModel(nn.Module):
    def __init__(self, num_layers: int, dim: int = 8, hidden: int = 16, rank: int = 2,
                 mlp_lora_rank: int = 0) -> None:
        super().__init__()
        self.embed = nn.Embedding(32, dim)
        self.blocks = nn.ModuleList(
            _FakeBlock(dim, hidden, rank, mlp_lora_rank) for _ in range(num_layers)
        )
        self.final_norm = nn.RMSNorm(dim)
        self.lm_head = _FakeLinear(32, dim)


def _cfg(num_layers: int = 4, fast_blocks: int = 2, **kw) -> ModelConfig:
    return ModelConfig(
        vocab_size=32, hidden_size=8, intermediate_size=16, num_layers=num_layers,
        num_heads=2, num_kv_heads=1, window_size=8, chunk_size=4,
        fast_blocks=fast_blocks, lora=LoRAConfig(rank=2, alpha=4.0), **kw
    )


# --------------------------------------------------------------------------


def test_split_is_a_partition() -> None:
    cfg = _cfg(num_layers=4, fast_blocks=2)
    train_cfg = TrainConfig(seq_len=16, tokens_per_step=32)
    model = _FakeModel(num_layers=4)

    split = split_parameters(model, cfg, train_cfg)
    split.assert_disjoint_and_total(model)

    all_names = {n for n, _ in model.named_parameters()}
    assert set(split.fast) | set(split.slow) | set(split.frozen) == all_names
    assert not (set(split.fast) & set(split.slow))
    assert not (set(split.fast) & set(split.frozen))
    assert not (set(split.slow) & set(split.frozen))

    # Exactly the last fast_blocks blocks' MLP projection weights are fast.
    expected_fast = {
        f"blocks.{i}.mlp.{w}.weight" for i in (2, 3) for w in ("w1", "w2", "w3")
    }
    assert set(split.fast) == expected_fast

    # Slow = every LoRA factor, every norm gain, every inner_lr_log.
    assert "blocks.0.attn.wq.lora_A" in split.slow
    assert "blocks.3.attn.wo.lora_B" in split.slow
    assert "blocks.1.seq_norm.weight" in split.slow
    assert "blocks.1.mlp_norm.weight" in split.slow
    assert "final_norm.weight" in split.slow
    assert "blocks.0.inner_lr_log" in split.slow

    # Frozen = the pretrained base weights we never touch.
    assert "embed.weight" in split.frozen
    assert "lm_head.weight" in split.frozen
    assert "blocks.0.attn.wq.weight" in split.frozen
    assert "blocks.0.mlp.w1.weight" in split.frozen  # block 0 is below first_fast_layer
    assert "blocks.1.mlp.w3.weight" in split.frozen

    # The returned tensors are the LIVE parameters, not copies.
    params = dict(model.named_parameters())
    for group in (split.fast, split.slow, split.frozen):
        for name, tensor in group.items():
            assert tensor is params[name]


def test_requires_grad_flags() -> None:
    cfg = _cfg()
    train_cfg = TrainConfig(seq_len=16, tokens_per_step=32)
    model = _FakeModel(num_layers=4)
    # Poison the flags so the split has to actually set them.
    for p in model.parameters():
        p.requires_grad_(False)

    split = split_parameters(model, cfg, train_cfg)

    assert split.slow and split.fast and split.frozen
    for p in split.slow.values():
        assert p.requires_grad is True
    # The inner loop differentiates the chunk loss w.r.t. fast weights, so they
    # need requires_grad even though the OUTER optimizer does not own them.
    for p in split.fast.values():
        assert p.requires_grad is True
    for p in split.frozen.values():
        assert p.requires_grad is False


def test_fast_excludes_lora_of_mlp() -> None:
    """The subtle case: a LoRA-wrapped MLP projection.

    `blocks.3.mlp.w1.weight` is the fast weight (inner loop owns it), while
    `blocks.3.mlp.w1.lora_A/B` are slow adapters on the same projection.
    """
    cfg = ModelConfig(
        vocab_size=32, hidden_size=8, intermediate_size=16, num_layers=4,
        num_heads=2, num_kv_heads=1, window_size=8, chunk_size=4, fast_blocks=2,
        lora=LoRAConfig(rank=2, alpha=4.0, targets=("wq", "wk", "wv", "wo", "w1")),
    )
    train_cfg = TrainConfig(seq_len=16, tokens_per_step=32)

    assert is_fast_param("blocks.3.mlp.w1.weight", cfg) is True
    assert is_fast_param("blocks.3.mlp.w1.lora_A", cfg) is False
    assert is_fast_param("blocks.3.mlp.w1.lora_B", cfg) is False
    assert is_slow_param("blocks.3.mlp.w1.lora_A", train_cfg) is True
    assert is_slow_param("blocks.3.mlp.w1.weight", train_cfg) is False

    model = _FakeModel(num_layers=4, mlp_lora_rank=2)
    split = split_parameters(model, cfg, train_cfg)
    split.assert_disjoint_and_total(model)
    assert "blocks.3.mlp.w1.weight" in split.fast
    assert "blocks.3.mlp.w1.lora_A" in split.slow
    assert "blocks.3.mlp.w1.lora_B" in split.slow


def test_is_fast_param_block_range() -> None:
    cfg = _cfg(num_layers=16, fast_blocks=4)  # first_fast_layer == 12
    assert is_fast_param("blocks.12.mlp.w1.weight", cfg) is True
    assert is_fast_param("blocks.15.mlp.w3.weight", cfg) is True
    assert is_fast_param("blocks.11.mlp.w1.weight", cfg) is False
    assert is_fast_param("blocks.0.mlp.w2.weight", cfg) is False
    # Not an MLP projection.
    assert is_fast_param("blocks.15.attn.wq.weight", cfg) is False
    assert is_fast_param("blocks.15.seq_norm.weight", cfg) is False
    # Not inside a block at all.
    assert is_fast_param("lm_head.weight", cfg) is False
    assert is_fast_param("embed.weight", cfg) is False


def test_fast_blocks_zero_gives_no_fast_params() -> None:
    cfg = _cfg(num_layers=4, fast_blocks=0)
    train_cfg = TrainConfig(seq_len=16, tokens_per_step=32)
    model = _FakeModel(num_layers=4)
    split = split_parameters(model, cfg, train_cfg)
    split.assert_disjoint_and_total(model)
    assert split.fast == {}
    assert "blocks.3.mlp.w1.weight" in split.frozen


def test_is_slow_param_substring_match() -> None:
    train_cfg = TrainConfig(seq_len=16, tokens_per_step=32)
    assert is_slow_param("blocks.0.attn.wq.lora_A", train_cfg) is True
    assert is_slow_param("blocks.0.seq_norm.weight", train_cfg) is True
    assert is_slow_param("final_norm.weight", train_cfg) is True
    assert is_slow_param("blocks.0.inner_lr_log", train_cfg) is True
    assert is_slow_param("blocks.0.attn.wq.weight", train_cfg) is False
    assert is_slow_param("embed.weight", train_cfg) is False

    custom = TrainConfig(seq_len=16, tokens_per_step=32, slow_spec=("lora_A",))
    assert is_slow_param("blocks.0.attn.wq.lora_B", custom) is False


def test_overlapping_fast_and_slow_raises() -> None:
    """A spec that claims a fast weight as slow is a configuration error."""
    cfg = _cfg(num_layers=4, fast_blocks=2)
    bad = TrainConfig(seq_len=16, tokens_per_step=32, slow_spec=("lora_A", "mlp.w1.weight"))
    model = _FakeModel(num_layers=4)
    with pytest.raises(ValueError, match="both fast and slow"):
        split_parameters(model, cfg, bad)


def test_assert_disjoint_and_total_catches_a_bad_split() -> None:
    cfg = _cfg(num_layers=4, fast_blocks=2)
    train_cfg = TrainConfig(seq_len=16, tokens_per_step=32)
    model = _FakeModel(num_layers=4)
    good = split_parameters(model, cfg, train_cfg)

    missing = ParamSplit(fast=dict(good.fast), slow=dict(good.slow), frozen={})
    with pytest.raises(ValueError):
        missing.assert_disjoint_and_total(model)

    dup = ParamSplit(
        fast=dict(good.fast),
        slow={**good.slow, **good.fast},
        frozen=dict(good.frozen),
    )
    with pytest.raises(ValueError):
        dup.assert_disjoint_and_total(model)

    extra = ParamSplit(
        fast=dict(good.fast),
        slow=dict(good.slow),
        frozen={**good.frozen, "not.a.param": torch.zeros(1)},
    )
    with pytest.raises(ValueError):
        extra.assert_disjoint_and_total(model)


def test_paramsplit_is_frozen_dataclass() -> None:
    split = ParamSplit(fast={}, slow={}, frozen={})
    with pytest.raises(Exception):
        split.fast = {}  # type: ignore[misc]


# ---------------------------------------------------------------- arm D: the full-slow spec
def test_full_slow_spec_makes_every_non_fast_parameter_slow() -> None:
    """slow_spec=("**",) is arm D: every parameter the inner loop does not own is trained by
    the outer loop. It used to match NOTHING -- is_slow_param is a substring test and "**"
    is a substring of no parameter name -- so arm D had an empty slow set and could never
    have run."""
    cfg = _cfg(num_layers=4, fast_blocks=2)
    model = _FakeModel(num_layers=4)
    split = split_parameters(model, cfg, TrainConfig(seq_len=16, tokens_per_step=32, slow_spec=("**",)))
    split.assert_disjoint_and_total(model)

    default = split_parameters(_FakeModel(num_layers=4), cfg, TrainConfig(seq_len=16, tokens_per_step=32))
    assert set(split.fast) == set(default.fast), "the wildcard must not change which weights are fast"
    assert split.fast and split.slow, "both sets must be non-empty"
    assert not split.frozen, f"full-slow leaves nothing frozen, got {sorted(split.frozen)[:3]}"
    assert all(p.requires_grad for p in split.slow.values())


def test_full_slow_wildcard_cannot_be_mixed_with_patterns() -> None:
    """("**", "lora_A") is a contradiction in terms; refuse it where it is written."""
    with pytest.raises(AssertionError, match="must be the only entry"):
        TrainConfig(seq_len=16, tokens_per_step=32, slow_spec=("**", "lora_A"))


def test_outer_set_is_the_slow_set_unless_the_fast_init_is_trained():
    """ParamSplit.outer is what the outer optimizer owns: the slow set, plus the fast
    weights (their initial value W_0) only when TrainConfig.fast_init_trained is set."""
    fast = {"blocks.1.mlp.w1.weight": nn.Parameter(torch.zeros(2, 2))}
    slow = {"blocks.1.attn.wq.lora_A": nn.Parameter(torch.zeros(1, 2))}

    plain = ParamSplit(fast=fast, slow=slow, frozen={})
    trained = ParamSplit(fast=fast, slow=slow, frozen={}, fast_init_trained=True)

    assert plain.outer == slow
    assert trained.outer == {**slow, **fast}
    assert plain.counts()["outer"] == 2 and trained.counts()["outer"] == 6
