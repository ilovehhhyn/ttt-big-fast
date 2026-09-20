"""Fixed-length sequences sliced out of a flat token memmap.

Layout is the one written by :mod:`ttt.data.prepare`: ``<path>/<split>.bin`` is
a flat uint32 token stream (documents concatenated, each prefixed by BOS) and
``<path>/<split>.json`` carries its length and the special-token ids.

Item ``i`` is the window::

    window = tokens[i*L : (i+1)*L + 1]        # L = seq_len, length L + 1
    input_ids = window[:-1]                   # L positions
    targets   = window[1:]                    # the same positions, shifted by 1

The ``+1`` is the whole reason a sequence costs ``L + 1`` tokens rather than
``L``: position ``L-1`` needs a token to predict, and taking it from the next
window (instead of dropping it) keeps every position of every item supervised.
Hence ``len(dataset) == (num_tokens - 1) // L``: the last item must satisfy
``(i+1)*L + 1 <= num_tokens``.

Windows are cut at a fixed stride and ignore document boundaries, exactly as in
TTT-E2E, which is safe precisely because `prepare` discarded every document
shorter than the context length: a window can therefore contain at most one
document boundary's worth of BOS, and the fast weights are reset per sequence.
``loss_mask`` is 0 wherever the TARGET is BOS -- predicting the
start-of-document marker is not a language-modelling event (e2e masks it), and
its loss would otherwise be dominated by an arbitrary corpus-ordering choice.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ttt.data.prepare import BOS_TOKEN_ID, TOKEN_DTYPE


class TokenSequenceDataset(Dataset):
    """Fixed-length sequences from a flat token memmap (see the module docstring)."""

    def __init__(self, path: Path, split: str, seq_len: int) -> None:
        assert seq_len >= 1, f"seq_len must be >= 1, got {seq_len}"
        self.dir = Path(path)
        self.split = split
        self.seq_len = int(seq_len)
        self.bin_path = self.dir / f"{split}.bin"
        meta_path = self.dir / f"{split}.json"
        assert self.bin_path.exists(), f"missing token file {self.bin_path}"
        assert meta_path.exists(), f"missing sidecar {meta_path}"

        self.meta = json.loads(meta_path.read_text())
        self.num_tokens = int(self.meta["num_tokens"])
        self.dtype = np.dtype(self.meta.get("dtype", np.dtype(TOKEN_DTYPE).name))
        self.bos_token_id = int(self.meta.get("bos_token_id", BOS_TOKEN_ID))
        on_disk = self.bin_path.stat().st_size // self.dtype.itemsize
        assert on_disk == self.num_tokens, (
            f"{self.bin_path} holds {on_disk} tokens but {meta_path.name} claims {self.num_tokens}"
        )
        # Opened lazily so that each DataLoader worker process gets its own
        # memmap instead of inheriting / pickling a parent's handle.
        self._tokens: np.memmap | None = None

    @property
    def tokens(self) -> np.memmap:
        if self._tokens is None:
            self._tokens = np.memmap(self.bin_path, dtype=self.dtype, mode="r")
        return self._tokens

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_tokens"] = None  # never ship a memmap across a process boundary
        return state

    def __len__(self) -> int:
        # item i needs tokens up to (i+1)*L inclusive -> (num_tokens - 1) // L items
        return max((self.num_tokens - 1) // self.seq_len, 0)

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        n = len(self)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(f"index {i} out of range for {n} sequences")
        start = i * self.seq_len
        window = np.asarray(self.tokens[start : start + self.seq_len + 1], dtype=np.int64)
        input_ids = torch.from_numpy(window[:-1].copy())
        targets = torch.from_numpy(window[1:].copy())
        # 0 where the target is BOS; see the module docstring.
        loss_mask = (targets != self.bos_token_id).to(torch.int64)
        return {"input_ids": input_ids, "targets": targets, "loss_mask": loss_mask}


def _shard_indices(num_items: int, *, shuffle: bool, seed: int, rank: int, world_size: int) -> list[int]:
    """Deterministic per-rank index shard.

    Every rank draws the SAME permutation from `seed` (so the shards are a
    partition of the epoch, disjoint and complete) and then takes the strided
    slice ``perm[rank::world_size]``.
    """
    assert world_size >= 1 and 0 <= rank < world_size
    if shuffle:
        g = torch.Generator().manual_seed(int(seed))
        order = torch.randperm(num_items, generator=g).tolist()
    else:
        order = list(range(num_items))
    return order[rank::world_size]


def build_dataloader(
    path: Path,
    split: str,
    seq_len: int,
    micro_batch: int,
    *,
    shuffle: bool,
    seed: int,
    num_workers: int = 2,
    rank: int = 0,
    world_size: int = 1,
    start_sequence: int = 0,
) -> DataLoader:
    """Deterministic shard-by-rank loader; the last partial batch is dropped.

    Dropping it keeps every optimiser step made of exactly
    ``micro_batch * world_size`` sequences, which is what the token budget
    accounting in the training loop assumes.

    ``start_sequence`` resumes an interrupted run. The training loop cycles this loader
    forever, so sequence n of a run is

        indices[n % N],   N = len(indices)

    and a run that has consumed ``start_sequence`` sequences must continue with
    indices[(start_sequence + j) % N] for j = 0, 1, ... Rotating the index list left by
    ``start_sequence % N`` and cycling THAT gives exactly this stream:

        rotated[j % N] = indices[(start_sequence + j % N) % N] = indices[(start_sequence + j) % N]
    """
    dataset = TokenSequenceDataset(path, split, seq_len)
    indices = _shard_indices(len(dataset), shuffle=shuffle, seed=seed, rank=rank, world_size=world_size)
    assert start_sequence >= 0, f"start_sequence must be >= 0, got {start_sequence}"
    if start_sequence > 0:
        # With micro_batch > 1, drop_last discards a partial batch at every epoch
        # boundary, so the cycled stream is no longer indices[n % N] and a rotation would
        # not reproduce it. The inner loop requires micro_batch == 1 anyway.
        assert micro_batch == 1, "start_sequence requires micro_batch == 1 (exact epoch boundary)"
        assert len(indices) > 0, "cannot resume into an empty shard"
        shift = start_sequence % len(indices)
        indices = indices[shift:] + indices[:shift]
    shard = torch.utils.data.Subset(dataset, indices)
    # shuffle is already baked into `indices`, so the sampler stays sequential:
    # the order is a pure function of (seed, rank, world_size).
    return DataLoader(
        shard,
        batch_size=micro_batch,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
