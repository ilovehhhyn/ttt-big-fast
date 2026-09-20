"""Render the 32K result tables as Markdown, straight from the result files.

Tables typed by hand drift from the files they summarise: a caption once claimed one inner
rule for three rows that did not share it. Every number printed here is read from a result
JSON, and every row states the settings it was run with, taken from that file's own args.

    python scripts/render_tables.py /scratch/gpfs/ARORA/hh9077/results
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(res: Path, name: str) -> dict | None:
    p = res / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def bands(token_nll: list[float], width: int) -> list[float]:
    return [sum(token_nll[s:s + width]) / len(token_nll[s:s + width]) for s in range(0, len(token_nll), width)]


def settings(r: dict) -> str:
    a = r["args"]
    inner = a["inner"] or {"A": "none", "B": "normalized_sgd"}.get(r["arm"], "?")
    lr = "-" if inner == "none" else f"{a['inner_lr']:g}"
    return f"T={a['seq_len']} k={a['window']} {inner} lr={lr} n={r['eval']['num_sequences']}"


def main() -> None:
    res = Path(sys.argv[1])

    print("### Arms A and B by token position (nothing trained)\n")
    print("| run | settings (from the file) | 0-8K | 8-16K | 16-24K | 24-32K | overall |")
    print("|---|---|---|---|---|---|---|")
    for name in ("A_32k_perseq", "A_32k_fullattn", "B_32k_perseq", "B_32k_fullattn"):
        r = load(res, name)
        if r is None:
            print(f"| {name} | MISSING | | | | | |")
            continue
        b = bands(r["eval"]["token_nll"], 8192)
        print(f"| {name} | {settings(r)} | " + " | ".join(f"{x:.4f}" for x in b) + f" | {r['eval']['loss']:.4f} |")

    print("\n### Matched context scaling: PG-19, normalized SGD at 4e-6, k = 8192\n")
    print("| T | T/k | arm A | arm B | B - A | source files |")
    print("|---|---|---|---|---|---|")
    for T, a_name, b_name in ((8192, "A_8k_pg19", "B_8k_pg19"), (32768, "A_32k_perseq", "B_32k_perseq")):
        ra, rb = load(res, a_name), load(res, b_name)
        if ra is None or rb is None:
            print(f"| {T} | | MISSING | | | {a_name}, {b_name} |")
            continue
        for r in (ra, rb):
            assert r["args"]["seq_len"] == T and r["args"]["window"] == 8192, (r["args"]["seq_len"], r["args"]["window"])
            assert "pg19" in r["args"]["data"], r["args"]["data"]
        assert rb["args"]["inner_lr"] == 4e-6 and (rb["args"]["inner"] or "normalized_sgd") == "normalized_sgd"
        print(f"| {T} | {T // 8192} | {ra['eval']['loss']:.4f} | {rb['eval']['loss']:.4f} | "
              f"{rb['eval']['loss'] - ra['eval']['loss']:+.4f} | {a_name}, {b_name} |")

    print("\n### AdamW as the inner optimizer at 32K (arm B, nothing trained)\n")
    ra = load(res, "A_32k_perseq")
    base = ra["eval"]["loss"] if ra else float("nan")
    print(f"| inner lr | loss | vs arm A ({base:.4f}) |")
    print("|---|---|---|")
    rows = []
    for p in res.glob("B_32k_adamw_lr*.json"):
        r = json.loads(p.read_text())
        assert r["args"]["inner"] == "adamw" and r["args"]["seq_len"] == 32768
        rows.append((r["args"]["inner_lr"], r["eval"]["loss"]))
    for lr, loss in sorted(rows):
        print(f"| {lr:g} | {loss:.4f} | {loss - base:+.4f} |")

    print("\n### What context beyond the window is worth (un-tuned model, full attention in both conditions)\n")
    print("| window S | recent context in the restart condition | full | restart | value (per document) | 95% CI | documents positive |")
    print("|---|---|---|---|---|---|---|")
    for S in (8192, 2048, 1024):
        r = load(res, f"context_value_32k_s{S}")
        if r is None:
            print(f"| {S} | MISSING | | | | | |")
            continue
        assert r["args"]["segment"] == S
        for q, b in sorted(r["bands"].items(), key=lambda kv: int(kv[0])):
            d = b["per_document"]
            print(f"| {S} | >= {int(q)} tokens | {b['full']:.4f} | {b['restart']:.4f} | {d['mean']:+.4f} | "
                  f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] | {d['positive']}/{d['n']} |")
        print(f"| {S} | sanity: segment 0, identical input | | | max abs diff {r['segment0_max_abs_difference']:.1e} | | |")


if __name__ == "__main__":
    main()
