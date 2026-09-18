"""Parameter-name conventions and the fast / slow / frozen partition.

Three disjoint roles, keyed purely off the parameter's dotted name:

fast    The MLP projection weights of the last ``cfg.fast_blocks`` blocks.
        The INNER loop rewrites these once per 1024-token chunk. The outer
        optimizer does not own them, but they still need ``requires_grad=True``
        because the inner loop takes d(chunk loss)/d(fast weight), and the
        outer loop differentiates through that.

slow    Everything the OUTER loop meta-learns: LoRA factors (``lora_A``,
        ``lora_B``), every RMSNorm gain (``norm.weight``), and the per-fast-tensor
        learned inner-learning-rate log-scalar (``inner_lr_log``). The list comes
        from ``TrainConfig.slow_spec`` and is matched as a SUBSTRING.

frozen  The pretrained base weights nobody updates: embeddings, lm_head,
        attention projection weights, and the MLP weights of blocks below
        ``cfg.first_fast_layer``.

Parameter names look like ``blocks.12.mlp.w1.weight``. A LoRA-wrapped MLP
projection is the subtle case: ``blocks.12.mlp.w1.weight`` is FAST (it is the
base matrix the inner loop rewrites) while ``blocks.12.mlp.w1.lora_A`` and
``.lora_B`` are SLOW adapters sitting on the same projection. The suffix test
below keys on ``.weight`` specifically, which is what keeps those apart.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from torch import Tensor, nn

from ttt.config import ModelConfig, TrainConfig

__all__ = [
    "FAST_SUFFIXES",
    "fast_suffixes",
    "ParamSplit",
    "is_fast_param",
    "is_slow_param",
    "split_parameters",
]

#: Parameter-name suffixes that make a tensor a fast weight. The ``.weight`` is
#: load-bearing: it selects the base matrix of an MLP projection and excludes
#: any ``lora_A`` / ``lora_B`` living on the very same projection.
FAST_SUFFIXES: tuple[str, ...] = (
    "mlp.w1.weight",
    "mlp.w2.weight",
    "mlp.w3.weight",
)


def fast_suffixes(cfg: ModelConfig) -> tuple[str, ...]:
    """Suffixes for THIS config's fast module.

    With cfg.prime the fast weight is the inserted ``mlp_prime``; the block's own
    ``mlp`` stays static as safe storage, so it must NOT be selected. Note that
    "blocks.i.mlp_prime.w1.weight" does not end with "mlp.w1.weight", so the two
    cases are disjoint and there is no accidental cross-match.
    """
    mod = cfg.fast_module
    return tuple(f"{mod}.{w}.weight" for w in ("w1", "w2", "w3"))

_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.")


def _block_index(name: str) -> int | None:
    """Block index for a block-scoped parameter name, else None."""
    match = _BLOCK_RE.match(name)
    return int(match.group(1)) if match else None


def is_fast_param(name: str, cfg: ModelConfig) -> bool:
    """True iff ``name`` is an MLP projection weight in one of the last
    ``cfg.fast_blocks`` blocks.

    Raises ValueError if the name carries a block index outside
    ``[0, cfg.num_layers)`` -- that means the model and the config disagree,
    which is a bug rather than something to silently classify as frozen.
    """
    index = _block_index(name)
    if index is None:
        return False
    if index >= cfg.num_layers:
        raise ValueError(
            f"parameter {name!r} names block {index}, but cfg.num_layers={cfg.num_layers}; "
            "model and config disagree"
        )
    if index < cfg.first_fast_layer:
        return False
    return name.endswith(fast_suffixes(cfg))


def is_slow_param(name: str, train_cfg: TrainConfig) -> bool:
    """True iff ``name`` contains any entry of ``train_cfg.slow_spec`` as a substring.

    Default spec: ``('lora_A', 'lora_B', 'norm.weight', 'inner_lr_log')``.
    Substring (not suffix) matching is deliberate: ``norm.weight`` has to catch
    ``blocks.3.seq_norm.weight``, ``blocks.3.mlp_norm.weight`` and ``final_norm.weight``
    alike.
    """
    return any(pattern in name for pattern in train_cfg.slow_spec)


@dataclass(frozen=True)
class ParamSplit:
    """The three parameter groups. Values are the LIVE ``nn.Parameter`` objects
    from the model -- no detached or cloned copies are made here, so mutating a
    tensor in one of these dicts mutates the model."""

    fast: dict[str, Tensor]
    slow: dict[str, Tensor]
    frozen: dict[str, Tensor]

    def assert_disjoint_and_total(self, model: nn.Module) -> None:
        """Raise unless fast/slow/frozen partition ``model``'s parameters exactly."""
        groups = {"fast": self.fast, "slow": self.slow, "frozen": self.frozen}
        actual = {name for name, _ in model.named_parameters()}

        seen: dict[str, str] = {}
        for group_name, group in groups.items():
            for name in group:
                if name in seen:
                    raise ValueError(
                        f"parameter {name!r} appears in both {seen[name]!r} and {group_name!r}"
                    )
                seen[name] = group_name

        unknown = set(seen) - actual
        if unknown:
            raise ValueError(f"split contains names that are not model parameters: {sorted(unknown)}")
        missing = actual - set(seen)
        if missing:
            raise ValueError(f"model parameters missing from the split: {sorted(missing)}")

        params = dict(model.named_parameters())
        for group_name, group in groups.items():
            for name, tensor in group.items():
                if tensor is not params[name]:
                    raise ValueError(
                        f"{group_name}[{name!r}] is not the live model parameter"
                    )

    def counts(self) -> dict[str, int]:
        """Element counts per group, for logging the fast/slow/frozen budget."""
        return {
            "fast": sum(t.numel() for t in self.fast.values()),
            "slow": sum(t.numel() for t in self.slow.values()),
            "frozen": sum(t.numel() for t in self.frozen.values()),
        }


def split_parameters(model: nn.Module, cfg: ModelConfig, train_cfg: TrainConfig) -> ParamSplit:
    """Partition ``model``'s named parameters into fast / slow / frozen.

    Also sets ``requires_grad``: slow True, fast True (the inner loop needs
    d(loss)/d(fast weight) even though the outer optimizer does not own them),
    frozen False.

    A parameter that matches both the fast rule and ``train_cfg.slow_spec`` is a
    configuration error and raises -- there is no precedence order to fall back on.
    """
    fast: dict[str, Tensor] = {}
    slow: dict[str, Tensor] = {}
    frozen: dict[str, Tensor] = {}

    for name, param in model.named_parameters():
        fast_hit = is_fast_param(name, cfg)
        slow_hit = is_slow_param(name, train_cfg)
        if fast_hit and slow_hit:
            raise ValueError(
                f"parameter {name!r} is both fast and slow: it is an MLP projection weight "
                f"in a fast block and also matches train_cfg.slow_spec={train_cfg.slow_spec!r}. "
                "Fix the slow_spec; the two roles are mutually exclusive."
            )
        if fast_hit:
            param.requires_grad_(True)
            fast[name] = param
        elif slow_hit:
            param.requires_grad_(True)
            slow[name] = param
        else:
            param.requires_grad_(False)
            frozen[name] = param

    split = ParamSplit(fast=fast, slow=slow, frozen=frozen)
    split.assert_disjoint_and_total(model)
    return split
