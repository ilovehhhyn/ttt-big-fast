"""Reads and writes the shared key directions that preconditioned_sgd removes from a fast-weight update.

A fast matrix W [out, in] sees inputs k_t [in] (its "keys"). Over ordinary text the second moment

    C = E[k k^T]          [in, in]

has a few eigenvectors that carry most of its trace: directions every token shares. The basis of a
matrix is E [in, r], the r strongest eigenvectors of C, with orthonormal columns. One file holds the
basis of every fast matrix, keyed by the fast-weight parameter name ("blocks.12.mlp.w1.weight").
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import Tensor

__all__ = ["KEY_BASIS_VERSION", "ORTHONORMAL_ATOL", "load_key_basis", "save_key_basis", "validate_key_basis"]

KEY_BASIS_VERSION = 1
# Largest |E^T E - I| entry accepted. float32 eigenvectors of an 8192 x 8192 matrix reach about 1e-5.
ORTHONORMAL_ATOL = 1e-4
_FIELDS = {"version", "basis", "meta"}


def validate_key_basis(basis: dict[str, Tensor]) -> dict[str, Tensor]:
    """Return `basis` unchanged if every entry is a finite [in, r] matrix with orthonormal columns."""
    assert len(basis) > 0, "key basis holds no matrices; rerun scripts/key_basis.py"
    for name in sorted(basis):
        e = basis[name]
        assert isinstance(e, Tensor) and e.ndim == 2 and e.is_floating_point(), (
            f"key basis entry {name!r} must be a 2-D floating tensor [in, r], got {type(e).__name__} "
            f"with shape {tuple(getattr(e, 'shape', ()))}"
        )
        in_features, rank = e.shape
        assert 1 <= rank <= in_features, f"key basis entry {name!r} has shape {tuple(e.shape)}; requires 1 <= r <= in"
        assert torch.isfinite(e).all(), f"key basis entry {name!r} holds non-finite values; rerun scripts/key_basis.py"
        gram_error = (e.mT.double() @ e.double() - torch.eye(rank, dtype=torch.float64, device=e.device)).abs().max().item()
        assert gram_error <= ORTHONORMAL_ATOL, (
            f"key basis entry {name!r} must have orthonormal columns: max |E^T E - I| = {gram_error:.2e} "
            f"exceeds {ORTHONORMAL_ATOL:.0e}; rerun scripts/key_basis.py"
        )
    return basis


def save_key_basis(path: Path, basis: dict[str, Tensor], meta: dict) -> None:
    """Write a validated basis and its provenance atomically."""
    payload = {"version": KEY_BASIS_VERSION, "basis": {k: v.detach().cpu().contiguous() for k, v in validate_key_basis(basis).items()},
               "meta": meta}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_key_basis(path: Path) -> dict[str, Tensor]:
    """Parse a basis file into validated [in, r] matrices keyed by fast-weight name."""
    assert path.exists(), f"key basis file {path} does not exist; write it with scripts/key_basis.py"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(payload, dict) and set(payload) == _FIELDS, (
        f"key basis file {path} must hold exactly the fields {sorted(_FIELDS)}, got "
        f"{sorted(payload) if isinstance(payload, dict) else type(payload).__name__}"
    )
    assert payload["version"] == KEY_BASIS_VERSION, (
        f"key basis file {path} has version {payload['version']!r}; this code reads version {KEY_BASIS_VERSION}; "
        "rerun scripts/key_basis.py"
    )
    return validate_key_basis(payload["basis"])
