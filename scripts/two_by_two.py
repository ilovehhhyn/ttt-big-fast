"""The 2x2 behind H1: how the slow weights were trained x whether TTT runs at evaluation.

H1 says slow weights meta-learned THROUGH the inner loop capture the gain of test-time
training. Crossing {trained through the inner loop, plain fine-tune} with {TTT on, TTT off}
separates three things, per evaluation sequence i (losses: lower is better):

    TTT effect on weights w      T_w(i)  = off_w(i) - on_w(i)
    training effect at setting s M_s(i)  = plain_s(i) - meta_s(i)
    interaction                  I(i)    = T_meta(i) - T_plain(i) = M_on(i) - M_off(i)

The interaction is the quantity H1 is about: does training through the inner loop make
test-time training MORE useful? A training effect that is the same with TTT off as with it on
is a better set of slow weights, not a better use of fast weights. All effects are paired per
sequence and clustered by document (ttt/eval/paired.py).

Each input is a ttt.run result evaluated with --eval-ttt-off, so it holds both columns of
one row:

    python scripts/two_by_two.py --meta results/cell_meta_10.json --plain results/cell_plainft_10.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ttt.data.dataset import TokenSequenceDataset, _shard_indices
from ttt.eval.paired import clustered_paired_stats, sequence_documents


def _line(label: str, s: dict) -> str:
    return (f"{label:<46} {s['mean']:+.4f}  95% CI [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]  "
            f"t={s['t']:6.2f}  positive {s['positive']}/{s['n']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="slow weights trained through the inner loop")
    ap.add_argument("--plain", required=True, help="slow weights from a plain fine-tune (inner loop off)")
    a = ap.parse_args()
    meta, plain = json.load(open(a.meta)), json.load(open(a.plain))
    for r, name in ((meta, a.meta), (plain, a.plain)):
        assert "eval_ttt_off" in r, f"{name} lacks eval_ttt_off; evaluate it with --eval-ttt-off"
    # The four cells must score the same sequences under the same protocol and inner rule.
    for k in ("data", "seq_len", "window", "chunk", "seed", "eval_sequences", "inner", "inner_lr", "fast_blocks"):
        assert meta["args"][k] == plain["args"][k], f"the two files differ in {k}: {meta['args'][k]} vs {plain['args'][k]}"
    args = meta["args"]

    cell = {("meta", "on"): meta["eval"], ("meta", "off"): meta["eval_ttt_off"],
            ("plain", "on"): plain["eval"], ("plain", "off"): plain["eval_ttt_off"]}
    n = args["eval_sequences"]
    per = {k: np.asarray(v["per_sequence_loss"]) for k, v in cell.items()}
    assert all(len(v) == n for v in per.values()), "cells evaluated different numbers of sequences"

    ds = TokenSequenceDataset(Path(args["data"]), "val", args["seq_len"])
    order = _shard_indices(len(ds), shuffle=True, seed=args["seed"], rank=0, world_size=1)[:n]
    docs = sequence_documents(np.asarray(ds.tokens), ds.bos_token_id, args["seq_len"], order)

    for r, label in ((meta, "meta"), (plain, "plain")):
        src = r.get("loaded_slow")
        print(f"{label:>6} weights: {Path(src['path']).name if src else '(trained in this run)'}"
              + (f", step {src['step']}, trained with --inner {src['fingerprint']['inner']}" if src else ""))
    print(f"\n| slow weights | TTT on at eval | TTT off at eval |\n|---|---|---|")
    print(f"| trained through the inner loop | {cell[('meta','on')]['loss']:.4f} | {cell[('meta','off')]['loss']:.4f} |")
    print(f"| plain fine-tune | {cell[('plain','on')]['loss']:.4f} | {cell[('plain','off')]['loss']:.4f} |")
    print(f"\n{n} sequences from {len(set(docs))} documents; effects per document, positive = lower loss\n")

    t_meta = per[("meta", "off")] - per[("meta", "on")]
    t_plain = per[("plain", "off")] - per[("plain", "on")]
    m_on = per[("plain", "on")] - per[("meta", "on")]
    m_off = per[("plain", "off")] - per[("meta", "off")]
    stat = lambda d: clustered_paired_stats(d.tolist(), docs)
    print(_line("TTT at eval, on meta-learned weights", stat(t_meta)))
    print(_line("TTT at eval, on plain fine-tuned weights", stat(t_plain)))
    print(_line("training through the inner loop, TTT on", stat(m_on)))
    print(_line("training through the inner loop, TTT off", stat(m_off)))
    print(_line("INTERACTION (what H1 is about)", stat(t_meta - t_plain)))


if __name__ == "__main__":
    main()
