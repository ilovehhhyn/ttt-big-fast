"""What does test-time training contribute, on identical trained weights?

Reads a ttt.run result produced with --eval-ttt-off and reports the paired difference
(inner loop off - inner loop on) per sequence, per DOCUMENT, and by token position.
The per-document version is the one to quote: evaluation sequences that fall inside the
same book are not independent (see ttt/eval/paired.py).

    python scripts/paired_ttt_effect.py results/C_32k_abl.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ttt.data.dataset import TokenSequenceDataset, _shard_indices
from ttt.eval.paired import clustered_paired_stats, paired_stats, sequence_documents


def _line(label: str, s: dict) -> str:
    return (f"{label:<14} n={s['n']:<3} mean {s['mean']:+.4f}  se {s['se']:.4f}  t={s['t']:6.2f}  "
            f"95% CI [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]  positive {s['positive']}/{s['n']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    a = ap.parse_args()

    r = json.load(open(a.result))
    assert "eval_ttt_off" in r, f"{a.result} has no eval_ttt_off block; rerun with --eval-ttt-off"
    on, off, args = r["eval"], r["eval_ttt_off"], r["args"]
    assert on["num_sequences"] == off["num_sequences"], "on/off evaluated different sequence counts"
    n, seq_len = on["num_sequences"], args["seq_len"]
    diffs = [y - x for x, y in zip(on["per_sequence_loss"], off["per_sequence_loss"], strict=True)]

    # Reproduce the evaluation order from the same function the loader uses, so the
    # sequence -> document mapping cannot drift from what was actually evaluated.
    ds = TokenSequenceDataset(Path(args["data"]), "val", seq_len)
    order = _shard_indices(len(ds), shuffle=True, seed=args["seed"], rank=0, world_size=1)[:n]
    docs = sequence_documents(np.asarray(ds.tokens), ds.bos_token_id, seq_len, order)

    print(f"{Path(a.result).name}: steps={args['steps']} tokens/step={args['tokens_per_step']} "
          f"truncate_bptt={args['truncate_bptt']}")
    print(f"inner loop ON {on['loss']:.4f}   OFF (same weights) {off['loss']:.4f}   "
          f"difference {off['loss'] - on['loss']:+.4f} nats")
    print(f"{n} sequences drawn from {len(set(docs))} distinct documents "
          f"(largest share: {max(docs.count(d) for d in set(docs))} sequences from one document)")
    print(_line("per sequence", paired_stats(diffs)))
    print(_line("per document", clustered_paired_stats(diffs, docs)) + "   <- quote this one")

    window = args["window"]
    ta, tb = np.asarray(on["token_nll"]), np.asarray(off["token_nll"])
    print(f"by position (attention window = {window}):")
    for s in range(0, seq_len, window):
        e = min(s + window, seq_len)
        print(f"  tokens {s:>6}-{e:<6}  on {ta[s:e].mean():.4f}  off {tb[s:e].mean():.4f}  "
              f"difference {tb[s:e].mean() - ta[s:e].mean():+.4f}")


if __name__ == "__main__":
    main()
