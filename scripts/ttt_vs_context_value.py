"""Does a TTT gain follow what out-of-window context is WORTH, or how much the window DAMAGES?

Language-model loss confounds two things test-time training can do for a windowed model:
carry information forward (memory) and repair the damage the window does (repair). On a
corpus whose documents differ a lot in how much long-range context is worth (SlimPajama:
arXiv and code against books and web text), the two make different predictions per document:

    memory   the gain grows with   ceiling_i = NLL_restart - NLL_full   (scripts/context_value.py:
             what everything outside a window of S tokens is worth to a healthy model)
    repair   the gain grows with   damage_i  = NLL_windowed - (NLL_full + ceiling_i)
             how far the windowed model sits above a healthy model limited to the same window

Both are measured per sequence on the SAME validation sequences (the first N of the
evaluation order), averaged within documents, and entered together in

    gain_d = b_0 + b_ceiling * ceiling_d + b_damage * damage_d + e_d        (ttt/eval/paired.py: ols_fit)

so b_ceiling is "nats gained per nat of available long-range information, at equal damage".
healthy_i = NLL_full + ceiling_i is approximate: NLL_full averages over all T tokens and the
ceiling over the tight band of later segments; it is a covariate, not a headline number.

A ceiling coefficient can be produced by things that merely travel with the ceiling, so two
specifications that could undercut it are always printed next to the main one:

    + loss level   adds NLL_full_d: is "ceiling" standing in for "easy, repetitive text"?
    + domain       adds one indicator per source domain (at least MIN_DOMAIN_DOCS documents; the
                   largest domain is the baseline): does the coefficient survive WITHIN domains,
                   or is it a difference between arXiv/code and books/web?

    python scripts/ttt_vs_context_value.py --data DIR --context-value cv_skip0.json cv_skip24.json ... \\
        --untuned-a A.json --untuned-b B.json [--meta META.json --plain PLAIN.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ttt.data.dataset import TokenSequenceDataset, _shard_indices
from ttt.eval.paired import cluster_means, clustered_paired_stats, ols_fit


MIN_DOMAIN_DOCS = 5  # a source domain gets its own indicator only with at least this many documents


def _per_seq(result: dict, block: str, n: int) -> np.ndarray:
    assert block in result, f"result lacks '{block}' (evaluate with --eval-ttt-off for the OFF condition)"
    losses = result[block]["per_sequence_loss"]
    assert len(losses) >= n, f"'{block}' holds {len(losses)} sequences, fewer than the {n} analysed"
    return np.asarray(losses[:n], dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--context-value", nargs="+", required=True,
                    help="context_value.py reports in --skip-sequences order; together they must cover sequences [0, N)")
    ap.add_argument("--untuned-a", required=True, help="arm A (no TTT), un-tuned, at the window being studied")
    ap.add_argument("--untuned-b", required=True, help="arm B (TTT alone), un-tuned, same window")
    ap.add_argument("--meta", default=None, help="slow weights trained through the inner loop, evaluated with --eval-ttt-off")
    ap.add_argument("--plain", default=None, help="plain fine-tuned slow weights, evaluated with TTT via --load-slow --eval-ttt-off")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    assert (a.meta is None) == (a.plain is None), "pass --meta and --plain together, or neither"

    # ---- context value: concatenate the pieces and check they tile [0, N) of the evaluation order
    pieces = [json.load(open(f)) for f in a.context_value]
    seg, seed, T = (pieces[0]["args"][k] for k in ("segment", "seed", "seq_len"))
    idx, docs, ceiling, full = [], [], [], []
    for f, r in zip(a.context_value, pieces, strict=True):
        ra = r["args"]
        assert (ra["segment"], ra["seed"], ra["seq_len"], ra["only_label"]) == (seg, seed, T, None), f"{f}: settings differ"
        assert ra["skip_sequences"] == len(idx), f"{f} starts at {ra['skip_sequences']}, expected {len(idx)}: pass pieces in order"
        tight = r["bands"][str((7 * seg) // 8)]
        idx += r["sequence_indices"]; docs += r["documents"]
        ceiling += tight["differences"]; full += r["per_sequence"]["full"]
    n = len(idx)
    ds = TokenSequenceDataset(Path(a.data), "val", T)
    order = _shard_indices(len(ds), shuffle=True, seed=seed, rank=0, world_size=1)[:n]
    assert idx == order, "the context-value pieces do not score the first N sequences of the evaluation order"
    ceiling, full = np.asarray(ceiling), np.asarray(full)

    runs = {"A": json.load(open(a.untuned_a)), "B": json.load(open(a.untuned_b))}
    if a.meta:
        runs["meta"], runs["plain"] = json.load(open(a.meta)), json.load(open(a.plain))
    for name, r in runs.items():
        ra = r["args"]
        assert (Path(ra["data"]).name, ra["seq_len"], ra["seed"]) == (Path(a.data).name, T, seed), f"{name}: data/seq_len/seed differ"
        assert ra["window"] == seg, f"{name} was evaluated at window {ra['window']}, the context value at S = {seg}"
    A, B = _per_seq(runs["A"], "eval", n), _per_seq(runs["B"], "eval", n)

    labels_path = Path(a.data) / "val_docs.json"
    doc_labels = json.loads(labels_path.read_text())["labels"] if labels_path.exists() else None
    healthy = full + ceiling  # [n] approximate loss of a healthy model limited to the window

    # gain name -> (per-sequence gain, per-sequence damage of the weights the gain is measured on)
    gains = {"TTT alone, un-tuned (A - B)": (A - B, A - healthy)}
    if a.meta:
        m_on, m_off = _per_seq(runs["meta"], "eval", n), _per_seq(runs["meta"], "eval_ttt_off", n)
        p_on, p_off = _per_seq(runs["plain"], "eval", n), _per_seq(runs["plain"], "eval_ttt_off", n)
        gains["TTT at eval, meta-learned weights"] = (m_off - m_on, m_off - healthy)
        gains["TTT at eval, plain fine-tuned weights"] = (p_off - p_on, p_off - healthy)
        gains["INTERACTION (meta - plain)"] = ((m_off - m_on) - (p_off - p_on), p_off - healthy)
        gains["training through the inner loop, TTT on"] = (p_on - m_on, p_off - healthy)

    print(f"{n} sequences from {len(set(docs))} documents; window S = {seg}; nats per token")
    report = {"n_sequences": n, "n_documents": len(set(docs)), "segment": seg, "by_label": {}, "regressions": {}}

    # ---- 1. by source domain
    groups = {"ALL": list(range(n))}
    if doc_labels is not None:
        for lab in sorted({doc_labels[d] for d in docs}):
            groups[lab] = [i for i in range(n) if doc_labels[docs[i]] == lab]
    head = f"{'domain':<22}{'seqs':>5}{'docs':>5}{'full':>8}{'ceiling':>9}{'damage A':>10}"
    print("\n" + head + "".join(f"{('gain' + str(j)):>9}" for j in range(len(gains))))
    for j, g in enumerate(gains):
        print(f"    gain{j} = {g}")
    for lab, ii in groups.items():
        row = {"sequences": len(ii), "documents": len({docs[i] for i in ii}), "full": float(full[ii].mean()),
               "ceiling": float(ceiling[ii].mean()), "damage_untuned": float((A - healthy)[ii].mean()), "gains": {}}
        for g, (gain, _) in gains.items():
            entry = {"mean": float(gain[ii].mean())}
            if row["documents"] >= 2:
                entry["per_document"] = clustered_paired_stats(gain[ii].tolist(), [docs[i] for i in ii])
            row["gains"][g] = entry
        report["by_label"][lab] = row
        print(f"{lab:<22}{row['sequences']:>5}{row['documents']:>5}{row['full']:>8.4f}{row['ceiling']:>+9.4f}"
              f"{row['damage_untuned']:>+10.4f}" + "".join(f"{v['mean']:>+9.4f}" for v in row["gains"].values()))

    # ---- 2. across documents: which quantity does each gain follow?
    print("\nacross documents: gain = b0 + b_ceiling * ceiling + b_damage * damage   (95% CI)")
    c_doc = cluster_means(ceiling.tolist(), docs)
    extra = {"+ loss level": {"full": cluster_means(full.tolist(), docs)}}
    if doc_labels is not None:
        doc_ids = sorted(set(docs))  # cluster_means returns clusters in sorted order
        counts = {lab: sum(doc_labels[d] == lab for d in doc_ids) for lab in {doc_labels[d] for d in doc_ids}}
        kept = sorted((lab for lab, c in counts.items() if c >= MIN_DOMAIN_DOCS), key=lambda lab: -counts[lab])
        # Baseline = the largest domain together with every domain too small for its own indicator.
        extra["+ domain"] = {f"is_{lab}": [float(doc_labels[d] == lab) for d in doc_ids] for lab in kept[1:]}
    for g, (gain, damage) in gains.items():
        base = {"ceiling": c_doc, "damage": cluster_means(damage.tolist(), docs)}
        y = cluster_means(gain.tolist(), docs)
        report["regressions"][g] = {}
        print(f"  {g}")
        for spec, cols in {"ceiling + damage": {}, **extra}.items():
            fit = ols_fit({**base, **cols}, y)
            report["regressions"][g][spec] = fit
            co = fit["coef"]
            print(f"    {spec:<18} b_ceiling {co['ceiling']['b']:+.3f} [{co['ceiling']['ci95'][0]:+.3f}, {co['ceiling']['ci95'][1]:+.3f}]   "
                  f"b_damage {co['damage']['b']:+.3f} [{co['damage']['ci95'][0]:+.3f}, {co['damage']['ci95'][1]:+.3f}]   "
                  f"b_0 {co['intercept']['b']:+.3f}   R2 {fit['r2']:.2f}  (n={fit['n']})")
    r = float(np.corrcoef(c_doc, cluster_means((A - healthy).tolist(), docs))[0, 1])
    report["corr_ceiling_damage_untuned"] = r
    print(f"  correlation between ceiling and un-tuned damage across documents: {r:+.2f} "
          f"(near +/-1 would mean the two cannot be told apart)")

    if a.out:
        Path(a.out).write_text(json.dumps(report))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
