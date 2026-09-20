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
OPERATIONAL_ARGS = frozenset({"out", "ckpt", "device", "hf_cache", "eval_sequences", "eval_ttt_off",
                              "forgetting_probe_tokens", "load_slow", "ckpt_every"})

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
    defaults: dict,
) -> tuple[int, list[dict]]:
    """Restore slow weights and optimizer state in place. Returns (next_step, history).

    `defaults` maps each current setting to its default value. It resolves one case the
    fingerprint alone cannot: a setting that did not exist when the checkpoint was written.
    Such a setting cannot have influenced that run, so the resume is the same experiment
    exactly when the setting is still at its default. (Jobs wait days in the queue while
    the code moves on; without this rule every new flag would strand their checkpoints.)
    A setting the checkpoint has but the code no longer knows is always refused.
    """
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    assert blob["format"] == _FORMAT, f"checkpoint format {blob['format']} != {_FORMAT}"

    # Same experiment? Name every differing setting: a resume under changed settings
    # would splice two experiments into one result.
    saved = blob["fingerprint"]
    differing = {}
    for k in sorted(set(saved) | set(fingerprint)):
        if k not in saved:
            # Added since the checkpoint: acceptable only at its default.
            assert k in defaults, f"no default known for setting {k!r}, which the checkpoint predates"
            if fingerprint[k] != defaults[k]:
                differing[k] = ("<did not exist>", fingerprint[k])
        elif k not in fingerprint:
            differing[k] = (saved[k], "<no longer exists>")
        elif saved[k] != fingerprint[k]:
            differing[k] = (saved[k], fingerprint[k])
    assert not differing, f"checkpoint was written under different settings (saved, now): {differing}"

    _assert_same_slow_set(blob["slow"], split)

    # Optimizer state is matched to parameters by POSITION inside each group, so the
    # groups must list the same names in the same order or the moments would be applied
    # to the wrong tensors without any error.
    assert _group_names(blob["optimizer"]) == _group_names(optimizer.state_dict()), (
        "optimizer param groups differ between checkpoint and model"
    )

    _copy_slow(blob["slow"], split)
    optimizer.load_state_dict(blob["optimizer"])
    return int(blob["step"]), list(blob["history"])


def _assert_same_slow_set(saved: dict, split: ParamSplit) -> None:
    """Same parameter names, same shapes: weights are only meaningful in the architecture
    and slow set they were trained in."""
    assert set(saved) == set(split.slow), (
        "slow parameter names differ between checkpoint and model: "
        f"only in checkpoint {sorted(set(saved) - set(split.slow))[:5]}, "
        f"only in model {sorted(set(split.slow) - set(saved))[:5]}"
    )
    for k, v in saved.items():
        assert v.shape == split.slow[k].shape, f"{k}: checkpoint {tuple(v.shape)} vs model {tuple(split.slow[k].shape)}"


def _copy_slow(saved: dict, split: ParamSplit) -> None:
    with torch.no_grad():
        for k, v in saved.items():
            split.slow[k].copy_(v)  # copy_ casts to the parameter's device and dtype


def load_slow_weights(path: Path, *, split: ParamSplit) -> dict:
    """Load ONLY the trained slow weights, for evaluation. Returns their provenance.

    This is not a resume. A resume must continue the same experiment, so it demands
    identical settings. Evaluating trained weights under a different inner rule is a
    different, legitimate question -- e.g. slow weights trained WITHOUT the inner loop,
    evaluated WITH it -- so no settings are compared here. The architecture and slow set
    must still match, and the checkpoint's step and settings are returned so the result
    file records exactly which weights were evaluated.
    """
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    assert blob["format"] == _FORMAT, f"checkpoint format {blob['format']} != {_FORMAT}"
    _assert_same_slow_set(blob["slow"], split)
    _copy_slow(blob["slow"], split)
    return {"path": str(path), "step": int(blob["step"]), "fingerprint": dict(blob["fingerprint"])}
