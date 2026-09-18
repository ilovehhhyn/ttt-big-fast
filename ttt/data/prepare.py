"""Corpus preparation: stream an HF dataset, tokenise it, write a flat token file.

We reproduce the data protocol of TTT-E2E (arXiv:2512.23675) but from free
Hugging Face sources instead of the paper's requester-pays GCS buckets:

  stage 1 (8K):  ``mlfoundations/dclm-baseline-1.0-parquet``, documents with
                 >= 8193 Llama-3 tokens.
  stage 2 (32K): ``deepmind/pg19``, books with >= 32769 tokens.

Why the length filter: the paper keeps only documents at least as long as the
context window (their footnote 4) so a training sequence never straddles a
document boundary in a way that would force the fast weights to be reset in the
middle of a sequence.  ``min_doc_tokens = seq_len + 1`` because a sequence of
``seq_len`` positions needs ``seq_len + 1`` tokens: ``seq_len`` inputs plus the
one extra token that is the target of the last position.

Storage format (one directory per corpus, one pair of files per split)::

    <out_dir>/<split>.bin   flat uint32 array, documents concatenated,
                            each document preceded by BOS (128000)
    <out_dir>/<split>.json  {"dtype", "num_tokens", "bos_token_id",
                             "eos_token_id", "tokenizer", "min_doc_tokens",
                             "num_docs", "source"}

This is e2e's idea (one flat array per split, sliced at fixed stride) with a
plain numpy memmap instead of zarr: the access pattern is a contiguous slice of
``seq_len + 1`` tokens, for which a memmap is both simpler and faster.

Tokenizer: ``meta-llama/Llama-3.2-1B`` is the canonical repo but it is GATED --
if the HF token in use has not been granted access it returns 403.  Pass
``--tokenizer-id unsloth/Llama-3.2-1B`` for the ungated mirror; its vocabulary
and special-token ids are identical (BOS 128000, EOS 128001), so the produced
.bin files are byte-for-byte the same.

Run ``python -m ttt.data.prepare --help`` for the CLI.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

# Llama-3 special tokens. Hard-coded because the whole pipeline is Llama-3 only
# and the ids must match what the model's embedding table expects.
BOS_TOKEN_ID = 128000  # Llama-3 <|begin_of_text|>; DEFAULT ONLY.
# The document separator must come from the tokenizer actually in use. Hardcoding the
# Llama-3 id writes token 128000 into, say, a SmolLM2 stream whose vocab is 49152, i.e.
# an out-of-range id that the embedding lookup cannot resolve. `resolve_bos_id` below is
# the single source of truth; BOS_TOKEN_ID remains as the fallback for tokenizers that
# declare no BOS.


def resolve_bos_id(tokenizer) -> int:
    """BOS id for `tokenizer`, falling back to eos, then to the Llama-3 default."""
    for attr in ("bos_token_id", "eos_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            return int(tid)
    return BOS_TOKEN_ID
EOS_TOKEN_ID = 128001  # <|end_of_text|>

TOKEN_DTYPE = np.uint32  # vocab is 128256 ids; uint32 halves the bytes of int64

_TOKENIZE_BATCH = 256  # texts per fast-tokenizer call
_LOG_EVERY_DOCS = 10_000  # documents *seen* between progress lines


@dataclass(frozen=True)
class PrepareSpec:
    """Everything that defines one prepared corpus."""

    dataset: str  # HF dataset id, e.g. "mlfoundations/dclm-baseline-1.0-parquet"
    split: str  # HF split to stream, e.g. "train"
    out_dir: Path
    out_split: str  # "train" or "val" -- the split this run's TRAIN half is named
    min_doc_tokens: int
    target_tokens: int  # stop once this many tokens have been written to out_split
    text_field: str = "text"
    tokenizer_id: str = "meta-llama/Llama-3.2-1B"
    val_every: int = 200  # every val_every-th KEPT document goes to val instead of train

    def __post_init__(self) -> None:
        assert self.min_doc_tokens >= 1
        assert self.target_tokens >= 1
        assert self.val_every >= 1
        assert self.out_split in ("train", "val"), f"out_split must be train/val, got {self.out_split!r}"

    @property
    def held_out_split(self) -> str:
        """Name of the companion split written alongside ``out_split``."""
        return "val" if self.out_split == "train" else "train"


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
class _BinWriter:
    """Append-stream token ids into ``<out_dir>/<split>.bin``.

    Kept separate from :func:`write_tokens` because :func:`prepare` consumes a
    single pass over the source stream and has to write BOTH splits at once;
    it therefore needs two live writers rather than two calls to a function
    that owns the whole iteration.
    """

    def __init__(self, out_dir: Path, split: str, bos_id: int = BOS_TOKEN_ID) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir
        self.split = split
        self.bos_id = int(bos_id)
        self.path = out_dir / f"{split}.bin"
        # "wb": a prepare run always produces the file from scratch, so a
        # re-run after a crash cannot silently append to a half-written corpus.
        self._fh = self.path.open("wb")
        self.num_tokens = 0
        self.num_docs = 0

    def add(self, tokens: list[int]) -> int:
        """Write ``[BOS] + tokens``; return how many tokens that added."""
        arr = np.empty(len(tokens) + 1, dtype=TOKEN_DTYPE)
        arr[0] = self.bos_id
        arr[1:] = np.asarray(tokens, dtype=TOKEN_DTYPE)
        arr.tofile(self._fh)
        self.num_tokens += arr.size
        self.num_docs += 1
        return arr.size

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> _BinWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def finalize(self, meta: dict) -> dict:
        """Write the sidecar json and return the metadata actually stored."""
        self.close()
        # The writer is the authority on which separator it actually wrote, so the
        # sidecar records self.bos_id unless the caller explicitly overrides it.
        meta = {"bos_token_id": self.bos_id, **meta}
        return write_meta(self.out_dir, self.split, meta, num_tokens=self.num_tokens, num_docs=self.num_docs)


def write_meta(out_dir: Path, split: str, meta: dict, *, num_tokens: int, num_docs: int) -> dict:
    """Write ``<out_dir>/<split>.json``. Counts are authoritative, not from `meta`."""
    full = {
        "dtype": np.dtype(TOKEN_DTYPE).name,
        "num_tokens": int(num_tokens),
        "bos_token_id": int(meta.get("bos_token_id", BOS_TOKEN_ID)),
        "eos_token_id": EOS_TOKEN_ID,
        "tokenizer": meta.get("tokenizer"),
        "min_doc_tokens": meta.get("min_doc_tokens"),
        "num_docs": int(num_docs),
        "source": meta.get("source"),
        **{k: v for k, v in meta.items() if k not in ("dtype", "num_tokens", "num_docs")},
    }
    (out_dir / f"{split}.json").write_text(json.dumps(full, indent=2) + "\n")
    return full


def write_tokens(out_dir: Path, split: str, docs: Iterable[list[int]], meta: dict) -> int:
    """Append-stream documents into ``<out_dir>/<split>.bin``, prefixing each with BOS.

    Returns the number of tokens written (BOS included) and writes the sidecar
    json.  ``meta`` supplies the descriptive fields (tokenizer, min_doc_tokens,
    source); the counts in the sidecar are computed here.
    """
    writer = _BinWriter(Path(out_dir), split)
    for tokens in docs:
        writer.add(list(tokens))
    writer.finalize(dict(meta))
    return writer.num_tokens


# ---------------------------------------------------------------------------
# the pure part: filtering + train/val assignment
# ---------------------------------------------------------------------------
def filter_and_split(
    docs: Iterable[list[int]],
    *,
    min_doc_tokens: int,
    val_every: int,
) -> Iterator[tuple[str, list[int]]]:
    """Drop short documents and assign the survivors to "train" or "val".

    Yields ``(split_name, token_ids)`` for every KEPT document, in source order.

    Rule (deterministic, and the reason the two splits are disjoint *by
    construction* rather than by a random draw): documents are numbered by
    their index among the KEPT documents only, and kept document ``k`` goes to
    val iff ``k % val_every == 0``.  Numbering by kept index rather than by
    source index keeps the val fraction at exactly 1/val_every regardless of
    how selective the length filter is, and depends on nothing but the order of
    the source stream -- no RNG, no hashing, so re-running the same stream
    reproduces the same assignment exactly.

    Pure and network-free: this is the half of :func:`prepare` the tests drive
    with synthetic token ids.
    """
    assert min_doc_tokens >= 1 and val_every >= 1
    kept = 0
    for tokens in docs:
        if len(tokens) < min_doc_tokens:  # >= threshold is kept, see module docstring
            continue
        yield ("val" if kept % val_every == 0 else "train"), tokens
        kept += 1


# ---------------------------------------------------------------------------
# the impure part: streaming + tokenising
# ---------------------------------------------------------------------------
@dataclass
class _Stats:
    seen: int = 0
    kept: int = 0
    skipped: int = 0  # documents that failed to tokenise


def tokenize_stream(
    texts: Iterable[str],
    encode_batch: Callable[[list[str]], list[list[int]]],
    stats: _Stats,
    *,
    batch_size: int = _TOKENIZE_BATCH,
) -> Iterator[list[int]]:
    """Tokenise `texts` in batches (HF fast tokenizers take a list of strings).

    THE ONE DELIBERATE try/except IN THIS MODULE.  A single malformed row
    (broken unicode, a surrogate pair the tokenizer rejects) must not kill a
    multi-hour streaming job, so a batch that raises is retried one text at a
    time and the individual offenders are skipped and counted
    (``stats.skipped``, reported at the end of :func:`prepare`).  Nothing else
    in this module has a fallback path.
    """
    batch: list[str] = []

    def flush() -> Iterator[list[int]]:
        if not batch:
            return
        try:
            yield from encode_batch(batch)
        except Exception:  # noqa: BLE001 - see docstring
            for text in batch:
                try:
                    yield encode_batch([text])[0]
                except Exception:  # noqa: BLE001
                    stats.skipped += 1
        batch.clear()

    for text in texts:
        batch.append(text)
        if len(batch) >= batch_size:
            yield from flush()
    yield from flush()


def _hf_texts(spec: PrepareSpec) -> Iterator[str]:
    """Stream the raw text column. Streaming is mandatory: DCLM is ~100B+ tokens."""
    from datasets import load_dataset

    ds = load_dataset(spec.dataset, split=spec.split, streaming=True)
    for row in ds:
        yield row[spec.text_field]


def prepare(spec: PrepareSpec, *, progress: bool = True) -> dict:
    """Stream, tokenise, filter, split into train/val, and write both splits.

    Returns a metadata dict holding the two sidecars plus the yield statistics
    the length filter produced (long documents are a small fraction of DCLM, so
    docs-seen vs docs-kept is the number that tells the user how long the job
    will take).

    Stops as soon as ``spec.out_split`` holds ``spec.target_tokens`` tokens; the
    companion split simply gets whatever the ``val_every`` rule sent its way by
    then (~1/val_every of the kept documents).
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec.tokenizer_id)

    def encode_batch(texts: list[str]) -> list[list[int]]:
        # add_special_tokens=False: BOS is added by the writer, exactly one per
        # document, so that the flat stream has no duplicated markers.
        return tokenizer(texts, add_special_tokens=False)["input_ids"]

    out_dir = Path(spec.out_dir)
    stats = _Stats()
    source = f"{spec.dataset}:{spec.split}"
    bos_id = resolve_bos_id(tokenizer)
    base_meta = {
        "bos_token_id": bos_id,
        "tokenizer": spec.tokenizer_id,
        "min_doc_tokens": spec.min_doc_tokens,
        "source": source,
        "dataset": spec.dataset,
        "hf_split": spec.split,
        "val_every": spec.val_every,
    }

    writers = {
        "train": _BinWriter(out_dir, "train", bos_id),
        "val": _BinWriter(out_dir, "val", bos_id),
    }
    started = time.time()
    last_log = 0

    def log(force: bool = False) -> None:
        nonlocal last_log
        if not progress:
            return
        if not force and stats.seen - last_log < _LOG_EVERY_DOCS:
            return
        last_log = stats.seen
        rate = 100.0 * stats.kept / max(stats.seen, 1)
        elapsed = time.time() - started
        print(
            f"[prepare] seen={stats.seen:,} kept={stats.kept:,} ({rate:.2f}%) "
            f"skipped={stats.skipped:,} train_tokens={writers['train'].num_tokens:,} "
            f"val_tokens={writers['val'].num_tokens:,} elapsed={elapsed / 60:.1f}min",
            file=sys.stderr,
            flush=True,
        )

    def counted(texts: Iterable[str]) -> Iterator[str]:
        for text in texts:
            stats.seen += 1
            log()
            yield text

    token_stream = tokenize_stream(counted(_hf_texts(spec)), encode_batch, stats)
    for split_name, tokens in filter_and_split(
        token_stream, min_doc_tokens=spec.min_doc_tokens, val_every=spec.val_every
    ):
        stats.kept += 1
        writers[split_name].add(tokens)
        if writers[spec.out_split].num_tokens >= spec.target_tokens:
            break
    log(force=True)

    meta = {
        name: writer.finalize({**base_meta, "split": name}) for name, writer in writers.items()
    }
    meta.update(
        docs_seen=stats.seen,
        docs_kept=stats.kept,
        docs_skipped=stats.skipped,
        keep_rate=stats.kept / max(stats.seen, 1),
        target_tokens=spec.target_tokens,
        out_split=spec.out_split,
        **{k: v for k, v in base_meta.items()},
    )
    if progress:
        print(
            f"[prepare] done: train={meta['train']['num_tokens']:,} tokens "
            f"({meta['train']['num_docs']:,} docs), val={meta['val']['num_tokens']:,} tokens "
            f"({meta['val']['num_docs']:,} docs); {stats.skipped:,} documents skipped "
            f"(tokenizer errors) out of {stats.seen:,} seen",
            file=sys.stderr,
            flush=True,
        )
    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    """One flag per PrepareSpec field, so an sbatch script can drive everything."""
    p = argparse.ArgumentParser(
        prog="python -m ttt.data.prepare",
        description="Tokenise a streamed HF corpus into a flat uint32 token file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    defaults = {f.name: f.default for f in fields(PrepareSpec)}
    p.add_argument("--dataset", required=True, help="HF dataset id to stream")
    p.add_argument("--split", required=True, help="HF split to stream (e.g. train)")
    p.add_argument("--out-dir", required=True, type=Path, help="output directory for <split>.bin/.json")
    p.add_argument("--out-split", default="train", choices=("train", "val"),
                   help="which written split the token target applies to")
    p.add_argument("--min-doc-tokens", required=True, type=int,
                   help="keep documents with at least this many tokens (= seq_len + 1)")
    p.add_argument("--target-tokens", required=True, type=float,
                   help="stop once out-split holds this many tokens (accepts 2e9)")
    p.add_argument("--text-field", default=defaults["text_field"], help="column holding the document text")
    p.add_argument("--tokenizer-id", default=defaults["tokenizer_id"],
                   help="HF tokenizer repo id (use unsloth/Llama-3.2-1B if the gated meta-llama repo 403s)")
    p.add_argument("--val-every", default=defaults["val_every"], type=int,
                   help="every N-th kept document goes to val")
    p.add_argument("--no-progress", action="store_true", help="silence the progress log")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    spec = PrepareSpec(
        dataset=args.dataset,
        split=args.split,
        out_dir=args.out_dir,
        out_split=args.out_split,
        min_doc_tokens=args.min_doc_tokens,
        target_tokens=int(args.target_tokens),
        text_field=args.text_field,
        tokenizer_id=args.tokenizer_id,
        val_every=args.val_every,
    )
    meta = prepare(spec, progress=not args.no_progress)
    print(json.dumps({k: v for k, v in meta.items()}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
