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


# ---------------------------------------------------------------------------
# resume: start_sequence continues the SAME infinite stream
# ---------------------------------------------------------------------------
def _cycled(dl, n: int) -> list[list[int]]:
    """First n sequences of the infinitely cycled loader, as run.py consumes it."""
    out: list[list[int]] = []
    while len(out) < n:
        for b in dl:
            out.extend(row.tolist() for row in b["input_ids"])
    return out[:n]


@pytest.mark.parametrize("start", [0, 1, 5, 7, 8, 9, 19])
def test_start_sequence_continues_the_same_stream(tmp_path, start: int) -> None:
    """A run resumed after consuming `start` sequences must see sequence start, start+1, ...

    of the ORIGINAL stream -- neither replaying data it already trained on nor skipping
    any. The cases cover mid-epoch (1, 5), the last item (7), exactly one epoch (8), just
    past it (9) and more than two epochs (19), since run.py cycles the loader forever.
    """
    _write_stream(tmp_path, "train", num_tokens=33)  # 8 items of seq_len 4
    take = 20

    def loader(s: int):
        return build_dataloader(tmp_path, "train", 4, 1, shuffle=True, seed=3,
                                num_workers=0, start_sequence=s)

    uninterrupted = _cycled(loader(0), start + take)
    assert _cycled(loader(start), take) == uninterrupted[start:], f"stream diverges at start={start}"


def test_start_sequence_requires_micro_batch_one(tmp_path) -> None:
    """With micro_batch > 1 the dropped partial batch makes the epoch boundary inexact,
    so an offset stream would no longer be a suffix of the original one."""
    _write_stream(tmp_path, "train", num_tokens=33)
    with pytest.raises(AssertionError, match="micro_batch == 1"):
        build_dataloader(tmp_path, "train", 4, 3, shuffle=True, seed=0, num_workers=0, start_sequence=2)


# ---------------------------------------------------------------------------
# labelled documents and the exact byte-length prefilter (SlimPajama)
# ---------------------------------------------------------------------------
def test_byte_prefilter_is_exact_never_dropping_a_document_that_could_qualify() -> None:
    """A token covers at least one UTF-8 byte, so a text with fewer bytes than
    min_doc_tokens cannot reach min_doc_tokens tokens. The filter must use BYTES, not
    characters: one rare character can become several byte-level tokens."""
    from ttt.data.prepare import may_reach_token_count

    assert not may_reach_token_count("a" * 9, 10)          # 9 bytes < 10 tokens: impossible
    assert may_reach_token_count("a" * 10, 10)             # 10 bytes: possible
    # 3 characters but 12 bytes: up to 12 byte-level tokens, so it may qualify for 10.
    assert may_reach_token_count("\U0001F600" * 3, 10)
    assert not may_reach_token_count("\U0001F600" * 2, 10)  # 8 bytes


def test_labels_stay_aligned_with_documents_even_when_one_fails_to_tokenise() -> None:
    """The label of document k must end up next to document k's tokens. A document that
    fails to tokenise is dropped WITH its label, not leaving later labels shifted by one."""
    from ttt.data.prepare import _Stats, tokenize_labeled_stream

    def encode_batch(texts: list[str]) -> list[list[int]]:
        if any(t == "BAD" for t in texts):
            raise ValueError("tokenizer rejected a text")
        return [[len(t)] * len(t) for t in texts]

    items = [("books", "aa"), ("arxiv", "BAD"), ("github", "cccc"), ("books", "b")]
    stats = _Stats()
    out = list(tokenize_labeled_stream(iter(items), encode_batch, stats, batch_size=3))
    assert out == [("books", [2, 2]), ("github", [4, 4, 4, 4]), ("books", [1])]
    assert stats.skipped == 1


def test_filter_and_split_labeled_matches_the_unlabelled_assignment() -> None:
    """Carrying labels must not change WHICH documents go to val."""
    from ttt.data.prepare import filter_and_split_labeled

    docs = [[1] * n for n in (5, 1, 7, 9, 2, 6, 8)]
    labelled = [(f"d{i}", d) for i, d in enumerate(docs)]
    plain = list(filter_and_split(iter(docs), min_doc_tokens=5, val_every=2))
    with_labels = list(filter_and_split_labeled(iter(labelled), min_doc_tokens=5, val_every=2))
    assert [(s, t) for s, _, t in with_labels] == plain
    assert [l for _, l, _ in with_labels] == ["d0", "d2", "d3", "d5", "d6"]


def test_prepare_writes_labels_aligned_with_the_documents_on_disk(tmp_path, monkeypatch) -> None:
    """End to end through prepare(): the k-th BOS-delimited document of <split>.bin must be
    the document whose label is labels[k] in <split>_docs.json, with short documents dropped
    before tokenisation and the val/train rule unchanged."""
    import transformers

    from ttt.data import prepare as prep

    class FakeTokenizer:
        bos_token_id, eos_token_id = 1, 2

        def __call__(self, texts, add_special_tokens=False):
            # one token per character, id = 10 + the digit the text is made of
            return {"input_ids": [[10 + int(t[0])] * len(t) for t in texts]}

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_a, **_k: FakeTokenizer())
    # (label, text): text is a repeated digit so each document's tokens identify it.
    rows = [("books", "0" * 6), ("web", "1" * 2), ("arxiv", "2" * 8), ("github", "3" * 5),
            ("web", "4" * 3), ("books", "5" * 7)]
    monkeypatch.setattr(prep, "_hf_items", lambda spec: iter(rows))

    spec = prep.PrepareSpec(dataset="fake/ds", split="train", out_dir=tmp_path, out_split="train",
                            min_doc_tokens=5, target_tokens=10**9, val_every=2,
                            label_field="meta.set_name")
    meta = prep.prepare(spec, progress=False)

    assert meta["docs_seen"] == 6 and meta["docs_kept"] == 4
    assert meta["docs_too_short_untokenised"] == 2          # "11" and "444" never reached the tokenizer
    # kept order: books(0) arxiv(2) github(3) books(5); kept index 0 and 2 go to val.
    want = {"val": [("books", 10), ("github", 13)], "train": [("arxiv", 12), ("books", 15)]}
    for split, expected in want.items():
        labels = json.loads((tmp_path / f"{split}_docs.json").read_text())
        assert labels["label_field"] == "meta.set_name"
        toks = np.fromfile(tmp_path / f"{split}.bin", dtype=np.uint32).tolist()
        docs, cur = [], None
        for t in toks:                                       # split the stream at BOS
            if t == 1:
                cur = []; docs.append(cur)
            else:
                cur.append(t)
        assert len(docs) == len(labels["labels"]) == len(expected)
        for doc, label, (want_label, want_token) in zip(docs, labels["labels"], expected, strict=True):
            assert label == want_label and set(doc) == {want_token}, (split, label, doc[:3])


def test_local_parquet_files_are_read_in_sorted_order(tmp_path) -> None:
    """--data-files reads downloaded parquet shards offline (compute nodes have no internet).
    Shards must be read in SORTED name order: that is the order the Hub streams them in, and
    extending a corpus relies on documents arriving in the original order."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from ttt.data import prepare as prep

    # Written out of order on purpose; names sort 00000 < 00001 < 00002.
    for shard, texts in (("train-00002-of-00003", ["e", "f"]), ("train-00000-of-00003", ["a", "b"]),
                         ("train-00001-of-00003", ["c", "d"])):
        pq.write_table(pa.table({"text": texts, "meta": [{"set": "x"}] * 2}), tmp_path / f"{shard}.parquet")

    spec = prep.PrepareSpec(dataset="unused/when-data-files-is-set", split="train", out_dir=tmp_path / "out",
                            out_split="train", min_doc_tokens=1, target_tokens=10,
                            data_files=str(tmp_path / "train-*.parquet"), label_field="meta.set")
    assert list(prep._hf_items(spec)) == [("x", t) for t in "abcdef"]


def test_data_files_that_match_nothing_is_an_error(tmp_path) -> None:
    from ttt.data import prepare as prep

    spec = prep.PrepareSpec(dataset="unused", split="train", out_dir=tmp_path, out_split="train",
                            min_doc_tokens=1, target_tokens=10, data_files=str(tmp_path / "nothing-*.parquet"))
    with pytest.raises(AssertionError, match="matched no files"):
        list(prep._hf_items(spec))
