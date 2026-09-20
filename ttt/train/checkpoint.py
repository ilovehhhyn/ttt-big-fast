"""Exact checkpoint / resume for the outer loop.

Contract: a run that is killed and resumed is indistinguishable from one that was never
interrupted. That needs exactly five pieces of state, because nothing else persists
across outer steps (the fast weights are reset at every sequence, and the training path
contains no randomness):

    slow parameters        the only weights the outer loop writes
    optimizer state        AdamW's moments and step counts carry the optimisation history
    step                   the NEXT step to run; both LR schedules are pure functions of it
    history                per-step metrics already logged
    data position          not stored: sequence n of a run is indices[n % len(indices)],
                           so it is recomputed as step * seqs_per_step

Writes are atomic (temp file + rename), so a job killed mid-save leaves the previous
checkpoint intact. Loads are strict: a checkpoint is only accepted for the same
configuration and the same parameter set it was written for.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from ttt.model.naming import ParamSplit

# Arguments that do not influence the trained weights. Everything else must match for a
# resume to be the same experiment.
OPERATIONAL_ARGS = frozenset({"out", "ckpt", "device", "hf_cache", "eval_sequences", "eval_ttt_off"})

_FORMAT = 1


def training_fingerprint(args: dict) -> dict:
    """The subset of CLI arguments that determines the training trajectory."""
    return {k: v for k, v in sorted(args.items()) if k not in OPERATIONAL_ARGS}


def _group_names(optimizer_state: dict) -> list[list[str]]:
    """Parameter names per param group (build_param_groups stores them, sorted)."""
    return [list(g["names"]) for g in optimizer_state["param_groups"]]


def save_checkpoint(
    path: Path,
    *,
    step: int,
    split: ParamSplit,
    optimizer: torch.optim.Optimizer,
    history: list[dict],
    fingerprint: dict,
) -> None:
    """Atomically write the state needed to continue from outer step `step`."""
    assert step >= 0, f"step must be the next step to run (>= 0), got {step}"
    path = Path(path)
    blob = {
        "format": _FORMAT,
        "step": step,
        "slow": {k: v.detach().cpu() for k, v in split.slow.items()},
        "optimizer": optimizer.state_dict(),
        "history": list(history),
        "fingerprint": dict(fingerprint),
    }
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(blob, tmp)
        # os.replace is atomic on POSIX: readers see the old file or the new one, never
        # a partial write.
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def load_checkpoint(
    path: Path,
    *,
    split: ParamSplit,
    optimizer: torch.optim.Optimizer,
    fingerprint: dict,
) -> tuple[int, list[dict]]:
    """Restore slow weights and optimizer state in place. Returns (next_step, history)."""
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    assert blob["format"] == _FORMAT, f"checkpoint format {blob['format']} != {_FORMAT}"

    # Same experiment? Name every differing setting: a resume under changed settings
    # would splice two experiments into one result.
    saved = blob["fingerprint"]
    differing = {k: (saved.get(k), fingerprint.get(k))
                 for k in sorted(set(saved) | set(fingerprint)) if saved.get(k) != fingerprint.get(k)}
    assert not differing, f"checkpoint was written under different settings (saved, now): {differing}"

    # Same parameter set, same shapes?
    assert set(blob["slow"]) == set(split.slow), (
        "slow parameter names differ between checkpoint and model: "
        f"only in checkpoint {sorted(set(blob['slow']) - set(split.slow))[:5]}, "
        f"only in model {sorted(set(split.slow) - set(blob['slow']))[:5]}"
    )
    for k, v in blob["slow"].items():
        assert v.shape == split.slow[k].shape, f"{k}: checkpoint {tuple(v.shape)} vs model {tuple(split.slow[k].shape)}"

    # Optimizer state is matched to parameters by POSITION inside each group, so the
    # groups must list the same names in the same order or the moments would be applied
    # to the wrong tensors without any error.
    assert _group_names(blob["optimizer"]) == _group_names(optimizer.state_dict()), (
        "optimizer param groups differ between checkpoint and model"
    )

    with torch.no_grad():
        for k, v in blob["slow"].items():
            split.slow[k].copy_(v)  # copy_ casts to the parameter's device and dtype
    optimizer.load_state_dict(blob["optimizer"])
    return int(blob["step"]), list(blob["history"])
