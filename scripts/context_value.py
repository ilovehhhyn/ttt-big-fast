"""How much is context beyond the attention window actually worth on this benchmark?

This bounds what ANY out-of-window memory mechanism (test-time training included) can gain.
The same tokens are scored twice by the same un-tuned model, both times with a healthy
full-attention forward, so nothing is broken by a sliding window in either condition:

    FULL     one forward over the whole T-token sequence: the token at absolute position p
             sees all p previous tokens.
    RESTART  the sequence is cut into T/S independent segments of S tokens, each run as a
             fresh context (positions restart at 0): the token at within-segment position q
             sees only q previous tokens. No BOS is inserted: evaluation windows start
             mid-document in BOTH conditions, so context length is the only difference.

For a token at within-segment position q of segment j >= 1 (absolute p = j*S + q):

    value(q) = NLL_restart - NLL_full = what the extra j*S tokens of older context are worth
               to a model that already has q tokens of recent context.

Restricting to q >= S/2 compares "has S/2..S recent tokens" against "has everything", which
UPPER-bounds the value of context older than S: a sliding window of size S always has S
recent tokens, more than the restart condition has. Differences are paired per sequence and
clustered by document (ttt/eval/paired.py).

    python scripts/context_value.py --data DIR --out results/context_value.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ttt.config import Config, InnerConfig, OuterConfig, TrainConfig
from ttt.data.dataset import TokenSequenceDataset, _shard_indices
from ttt.eval.paired import clustered_paired_stats, paired_stats, sequence_documents
from ttt.model.naming import split_parameters
from ttt.optim.inner import build_inner_optimizer
from ttt.train.inner_loop import TTTInnerLoop
from ttt.utils.hf_import import MIRROR_REPO, build_llama_ttt


def _loop(model, seq_len: int, segment: int) -> tuple[TTTInnerLoop, object]:
    """An arm-A loop (no inner optimizer, nothing trained) for sequences of `seq_len`."""
    cfg = Config(model=model.cfg, inner=InnerConfig(optimizer="none", lr_warmup_frac=0.0),
                 outer=OuterConfig(lr=0.0, total_steps=1, warmup_frac=0.0),
                 train=TrainConfig(seq_len=seq_len, tokens_per_step=seq_len, remat_group=1,
                                   prefix_segment=segment, slow_spec=("__none__",), dtype="bf16"))
    split = split_parameters(model, model.cfg, cfg.train)
    return TTTInnerLoop(model, cfg, build_inner_optimizer(cfg.inner)), split


def _nll(loop: TTTInnerLoop, split, ids: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Per-token NLL [T] of one sequence. enable_grad is required by run_sequence's contract
    even though arm A takes no inner gradient."""
    with torch.enable_grad():
        out = loop.run_sequence(ids, tgt, mask, dict(split.fast), inference=True)
    return out.token_nll.float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=32768)
    ap.add_argument("--segment", type=int, default=8192, help="S: restart length = the window being bounded")
    ap.add_argument("--eval-sequences", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-sequences", type=int, default=0,
                    help="score sequences [skip, skip + eval-sequences) of the evaluation order (after --only-label), "
                         "so that a large sample can be scored as several short, disjoint runs")
    ap.add_argument("--only-label", default=None,
                    help="evaluate only sequences whose document carries this label (corpora prepared "
                         "with --label-field, e.g. RedPajamaArXiv), so each domain gets its own sample")
    a = ap.parse_args()
    T, S = a.seq_len, a.segment
    assert T % S == 0 and T // S >= 2, f"seq_len {T} must be a multiple (>= 2x) of segment {S}"

    dev = torch.device("cuda")
    # window = T: full attention for the whole sequence, and therefore also for every
    # S-token segment. One model serves both conditions.
    model = build_llama_ttt(MIRROR_REPO, max_seq_len=T, window_size=T, chunk_size=1024,
                            fast_blocks=4, lora=None, dtype=torch.float32).to(dev)
    model.remat_blocks = True
    full_loop, full_split = _loop(model, T, S)
    seg_loop, seg_split = _loop(model, S, S)

    ds = TokenSequenceDataset(Path(a.data), "val", T)
    order = _shard_indices(len(ds), shuffle=True, seed=a.seed, rank=0, world_size=1)
    docs = sequence_documents(np.asarray(ds.tokens), ds.bos_token_id, T, order)
    if a.only_label is not None:
        doc_labels = json.loads((Path(a.data) / "val_docs.json").read_text())["labels"]
        keep = [i for i, d in enumerate(docs) if doc_labels[d] == a.only_label]
        assert keep, f"no validation sequence has label {a.only_label!r}; labels present: {sorted(set(doc_labels))}"
        order, docs = [order[i] for i in keep], [docs[i] for i in keep]
    assert 0 <= a.skip_sequences < len(order), f"--skip-sequences {a.skip_sequences} leaves none of {len(order)} sequences"
    piece = slice(a.skip_sequences, a.skip_sequences + a.eval_sequences)
    order, docs = order[piece], docs[piece]
    assert len(set(docs)) >= 2, f"only {len(set(docs))} document(s) selected: a per-document interval needs at least 2"

    full = np.zeros((len(order), T), dtype=np.float64)
    restart = np.zeros((len(order), T), dtype=np.float64)
    for n, idx in enumerate(order):
        item = ds[idx]
        ids, tgt, mask = (item[k].unsqueeze(0).to(dev) for k in ("input_ids", "targets", "loss_mask"))
        full[n] = _nll(full_loop, full_split, ids, tgt, mask)
        for j in range(T // S):
            sl = slice(j * S, (j + 1) * S)
            restart[n, sl] = _nll(seg_loop, seg_split, ids[:, sl], tgt[:, sl], mask[:, sl])
        print(f"[{n + 1}/{len(order)}] full {full[n].mean():.4f}  restart {restart[n].mean():.4f}", flush=True)

    # Compare only tokens that (a) lie in segments j >= 1, so older context exists, and
    # (b) have q >= q_min recent tokens in the restart condition.
    # per_sequence: mean NLL over ALL T tokens of each sequence, so that other per-sequence
    # quantities (e.g. a windowed model's loss on the same sequence) can be set against it.
    report = {"args": vars(a), "documents": docs, "distinct_documents": len(set(docs)), "bands": {},
              "sequence_indices": [int(i) for i in order],
              "per_sequence": {"full": full.mean(axis=1).tolist(), "restart": restart.mean(axis=1).tolist()}}
    print(f"\n{len(order)} sequences from {len(set(docs))} documents. value = NLL_restart - NLL_full (nats/token)")
    for q_min in (S // 2, (3 * S) // 4, (7 * S) // 8):
        cols = np.concatenate([np.arange(j * S + q_min, (j + 1) * S) for j in range(1, T // S)])
        d = (restart[:, cols].mean(axis=1) - full[:, cols].mean(axis=1)).tolist()
        per_seq, per_doc = paired_stats(d), clustered_paired_stats(d, docs)
        report["bands"][str(q_min)] = {"recent_tokens_at_least": q_min, "per_sequence": per_seq,
                                       "per_document": per_doc, "differences": d,
                                       "full": float(full[:, cols].mean()), "restart": float(restart[:, cols].mean())}
        print(f"  recent context >= {q_min:>5}: full {full[:, cols].mean():.4f}  restart {restart[:, cols].mean():.4f}  "
              f"value {per_doc['mean']:+.4f}  per-document 95% CI [{per_doc['ci95'][0]:+.4f}, {per_doc['ci95'][1]:+.4f}] "
              f"(n={per_doc['n']}, positive {per_doc['positive']}/{per_doc['n']})")
    # Per-domain breakdown, when the corpus was prepared with --label-field (SlimPajama):
    # which KIND of text has long-range context worth remembering? Tightest band only.
    labels_path = Path(a.data) / "val_docs.json"
    if labels_path.exists():
        doc_labels = json.loads(labels_path.read_text())["labels"]
        seq_labels = [doc_labels[d] for d in docs]
        q_min = (7 * S) // 8
        cols = np.concatenate([np.arange(j * S + q_min, (j + 1) * S) for j in range(1, T // S)])
        d_all = restart[:, cols].mean(axis=1) - full[:, cols].mean(axis=1)
        report["by_label"] = {}
        print(f"  by label (recent context >= {q_min}):")
        for lab in sorted(set(seq_labels)):
            idx = [i for i, l in enumerate(seq_labels) if l == lab]
            n_docs = len({docs[i] for i in idx})
            entry = {"sequences": len(idx), "documents": n_docs, "full": float(full[idx][:, cols].mean()),
                     "restart": float(restart[idx][:, cols].mean()), "value": float(d_all[idx].mean())}
            if n_docs >= 2:   # an interval needs at least two documents
                entry["per_document"] = clustered_paired_stats(d_all[idx].tolist(), [docs[i] for i in idx])
            report["by_label"][lab] = entry
            ci = entry.get("per_document", {}).get("ci95")
            print(f"    {lab:<24} {len(idx):>3} seqs / {n_docs:>3} docs  full {entry['full']:.4f}  value {entry['value']:+.4f}"
                  + (f"  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "  (one document: no interval)"))

    # Sanity: in segment 0 the two conditions see IDENTICAL input, so they must agree up to
    # kernel noise. A large value here means the two code paths differ in more than context.
    seg0 = float(np.abs(restart[:, :S].mean(axis=1) - full[:, :S].mean(axis=1)).max())
    report["segment0_max_abs_difference"] = seg0
    print(f"  sanity, segment 0 (identical context): max |restart - full| per sequence = {seg0:.2e}")
    report["full_curve"] = full.mean(axis=0).tolist()
    report["restart_curve"] = restart.mean(axis=0).tolist()
    Path(a.out).write_text(json.dumps(report))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
