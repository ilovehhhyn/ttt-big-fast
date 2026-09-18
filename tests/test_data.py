"""Tests for ttt.data (Task 5): tokenised-corpus preparation and the sequence loader.

Every test here runs OFFLINE. Nothing touches Hugging Face: the streaming /
tokenising half of `prepare` is exercised through the pure generator
`filter_and_split`, which is fed synthetic (text, token_ids) pairs.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ttt.data.dataset import TokenSequenceDataset, build_dataloader
from ttt.data.prepare import BOS_TOKEN_ID, EOS_TOKEN_ID, filter_and_split, write_tokens

META = {"tokenizer": "test-tokenizer", "min_doc_tokens": 1, "source": "synthetic:train"}


def _write(out_dir, split: str, docs: list[list[int]], **meta) -> int:
    return write_tokens(out_dir, split, docs, {**META, **meta})


def _flat(out_dir, split: str) -> list[int]:
    """The raw on-disk token stream, read independently of the Dataset class."""
    return np.fromfile(out_dir / f"{split}.bin", dtype=np.uint32).tolist()


def _write_stream(out_dir, split: str, num_tokens: int) -> list[int]:
    """Write a single document so that the flat stream is exactly `num_tokens` long.

    The stream is [BOS, 1, 2, ..., num_tokens-1], i.e. position p holds value p
    for every p > 0, which makes off-by-one errors immediately visible.
    """
    doc = list(range(1, num_tokens))  # BOS is prepended by write_tokens
    written = _write(out_dir, split, [doc])
    assert written == num_tokens
    return [BOS_TOKEN_ID, *doc]


# ---------------------------------------------------------------------------
# 1. storage round-trip
# ---------------------------------------------------------------------------
def test_write_and_read_roundtrip(tmp_path) -> None:
    doc0, doc1, doc2 = [1, 2, 3], [4, 5], [6, 7, 8, 9]
    n = _write(tmp_path, "train", [doc0, doc1, doc2])

    expected = [BOS_TOKEN_ID, *doc0, BOS_TOKEN_ID, *doc1, BOS_TOKEN_ID, *doc2]
    assert n == len(expected) == 12
    assert _flat(tmp_path, "train") == expected

    meta = json.loads((tmp_path / "train.json").read_text())
    assert meta["dtype"] == "uint32"
    assert meta["num_tokens"] == 12
    assert meta["num_docs"] == 3
    assert meta["bos_token_id"] == BOS_TOKEN_ID == 128000
    assert meta["eos_token_id"] == EOS_TOKEN_ID == 128001
    assert meta["tokenizer"] == "test-tokenizer"
    assert meta["source"] == "synthetic:train"

    # Reading the same bytes back through the Dataset must reproduce the stream:
    # one item of seq_len=11 spans 11 inputs + 1 shifted target = all 12 tokens.
    ds = TokenSequenceDataset(tmp_path, "train", seq_len=11)
    assert len(ds) == 1
    item = ds[0]
    recovered = item["input_ids"].tolist() + [int(item["targets"][-1])]
    assert recovered == expected


# ---------------------------------------------------------------------------
# 2. shapes and the next-token shift
# ---------------------------------------------------------------------------
def test_dataset_shapes_and_shift(tmp_path) -> None:
    import torch

    flat = _write_stream(tmp_path, "train", num_tokens=33)
    seq_len = 4
    ds = TokenSequenceDataset(tmp_path, "train", seq_len=seq_len)
    assert len(ds) == 8

    for i in range(len(ds)):
        item = ds[i]
        x, y, m = item["input_ids"], item["targets"], item["loss_mask"]
        assert x.shape == (seq_len,) and y.shape == (seq_len,) and m.shape == (seq_len,)
        assert x.dtype == torch.int64 and y.dtype == torch.int64
        # item i is the window flat[i*L : (i+1)*L + 1]
        assert x.tolist() == flat[i * seq_len : (i + 1) * seq_len]
        assert y.tolist() == flat[i * seq_len + 1 : (i + 1) * seq_len + 1]
        # classic shift, checked inside the item
        for j in range(seq_len - 1):
            assert int(y[j]) == int(x[j + 1])


# ---------------------------------------------------------------------------
# 3. length formula
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "num_tokens, seq_len, expected",
    [
        (25, 8, 3),  # exact boundary: k*L + 1 with k = 3
        (24, 8, 2),  # one token short of the boundary
        (26, 8, 3),  # one token past it
        (9, 8, 1),
        (8, 8, 0),  # a full sequence but no target for the last position
    ],
)
def test_len_formula(tmp_path, num_tokens: int, seq_len: int, expected: int) -> None:
    _write_stream(tmp_path, "train", num_tokens=num_tokens)
    ds = TokenSequenceDataset(tmp_path, "train", seq_len=seq_len)
    assert len(ds) == (num_tokens - 1) // seq_len == expected


# ---------------------------------------------------------------------------
# 4. BOS targets are masked out of the loss
# ---------------------------------------------------------------------------
def test_loss_mask_masks_bos_targets(tmp_path) -> None:
    _write(tmp_path, "train", [[1, 2, 3], [4, 5, 6]])
    # stream: [BOS, 1, 2, 3, BOS, 4, 5, 6]  (BOS in the middle at index 4)
    ds = TokenSequenceDataset(tmp_path, "train", seq_len=7)
    item = ds[0]
    assert item["input_ids"].tolist() == [BOS_TOKEN_ID, 1, 2, 3, BOS_TOKEN_ID, 4, 5]
    assert item["targets"].tolist() == [1, 2, 3, BOS_TOKEN_ID, 4, 5, 6]
    assert item["loss_mask"].tolist() == [1, 1, 1, 0, 1, 1, 1]
    # the mask is 0 exactly where the TARGET is BOS, never where the input is
    mask = item["loss_mask"].bool()
    assert not (item["targets"][mask] == BOS_TOKEN_ID).any()
    assert int((~mask).sum()) == 1


# ---------------------------------------------------------------------------
# 5. deterministic, disjoint val split
# ---------------------------------------------------------------------------
def test_val_split_disjoint_and_deterministic() -> None:
    pairs = [(f"doc {i}", list(range(i, i + 10))) for i in range(10)]

    def run() -> list[tuple[str, tuple[int, ...]]]:
        docs = (ids for _text, ids in pairs)
        return [(split, tuple(ids)) for split, ids in filter_and_split(docs, min_doc_tokens=10, val_every=3)]

    first, second = run(), run()
    assert first == second, "filter_and_split must be deterministic"
    assert len(first) == 10

    # every val_every-th KEPT document (kept index 0, 3, 6, 9) goes to val
    assert [i for i, (s, _) in enumerate(first) if s == "val"] == [0, 3, 6, 9]
    train_docs = {ids for s, ids in first if s == "train"}
    val_docs = {ids for s, ids in first if s == "val"}
    assert not (train_docs & val_docs)
    assert len(train_docs) + len(val_docs) == 10
    assert {s for s, _ in first} == {"train", "val"}


# ---------------------------------------------------------------------------
# 6. minimum-length filter
# ---------------------------------------------------------------------------
def test_min_doc_filter() -> None:
    short = list(range(4))
    exact = list(range(5))  # exactly at the threshold -> KEPT
    long = list(range(6))
    out = list(filter_and_split(iter([short, exact, long]), min_doc_tokens=5, val_every=1_000_000))
    assert [ids for _s, ids in out] == [exact, long]
    assert all(s == "train" for s, _ in out[1:])
    assert all(len(ids) >= 5 for _s, ids in out)

    # nothing survives an impossible threshold
    assert list(filter_and_split(iter([short, exact, long]), min_doc_tokens=99, val_every=200)) == []


# ---------------------------------------------------------------------------
# 7. rank sharding
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shuffle", [False, True])
def test_dataloader_sharding(tmp_path, shuffle: bool) -> None:
    _write_stream(tmp_path, "train", num_tokens=33)
    seq_len, micro_batch = 4, 2
    full = TokenSequenceDataset(tmp_path, "train", seq_len=seq_len)
    assert len(full) == 8
    expected = {tuple(full[i]["input_ids"].tolist()) for i in range(len(full))}

    seen: list[set[tuple[int, ...]]] = []
    for rank in (0, 1):
        dl = build_dataloader(
            tmp_path, "train", seq_len, micro_batch,
            shuffle=shuffle, seed=1234, num_workers=0, rank=rank, world_size=2,
        )
        rows: set[tuple[int, ...]] = set()
        batches = 0
        for batch in dl:
            assert batch["input_ids"].shape == (micro_batch, seq_len)
            assert batch["targets"].shape == (micro_batch, seq_len)
            assert batch["loss_mask"].shape == (micro_batch, seq_len)
            batches += 1
            for row in batch["input_ids"]:
                rows.add(tuple(row.tolist()))
        assert batches == 2  # 8 items / 2 ranks / 2 per batch
        seen.append(rows)

    assert not (seen[0] & seen[1]), "ranks must see disjoint items"
    assert seen[0] | seen[1] == expected, "the two ranks together must cover the dataset"


def test_dataloader_drops_last_partial_batch(tmp_path) -> None:
    _write_stream(tmp_path, "train", num_tokens=33)  # 8 items of seq_len 4
    dl = build_dataloader(tmp_path, "train", 4, 3, shuffle=False, seed=0, num_workers=0)
    batches = list(dl)
    assert len(batches) == 2  # 8 // 3, the trailing 2 items are dropped
    assert all(b["input_ids"].shape == (3, 4) for b in batches)


def test_dataloader_is_deterministic_given_seed(tmp_path) -> None:
    _write_stream(tmp_path, "train", num_tokens=33)

    def order(seed: int) -> list[list[int]]:
        dl = build_dataloader(tmp_path, "train", 4, 2, shuffle=True, seed=seed, num_workers=0)
        return [row.tolist() for b in dl for row in b["input_ids"]]

    assert order(7) == order(7)
    assert order(7) != order(8)


def test_bos_id_comes_from_the_tokenizer_not_a_constant(tmp_path):
    """Regression: the separator id must follow the tokenizer.

    Hardcoding Llama-3's 128000 writes an out-of-range id into any corpus built with a
    smaller vocabulary (e.g. SmolLM2's 49152), which the embedding lookup cannot resolve.
    """
    from ttt.data.prepare import BOS_TOKEN_ID, _BinWriter, resolve_bos_id

    class Tok:
        bos_token_id = 7
        eos_token_id = 9

    class TokNoBos:
        bos_token_id = None
        eos_token_id = 11

    class TokNeither:
        bos_token_id = None
        eos_token_id = None

    assert resolve_bos_id(Tok()) == 7
    assert resolve_bos_id(TokNoBos()) == 11, "must fall back to eos"
    assert resolve_bos_id(TokNeither()) == BOS_TOKEN_ID, "then to the documented default"

    import numpy as np

    w = _BinWriter(tmp_path, "train", bos_id=7)
    w.add([1, 2, 3])
    w.close()
    arr = np.fromfile(tmp_path / "train.bin", dtype=np.uint32)
    assert arr[0] == 7, f"writer used {arr[0]}, not the tokenizer's BOS"
