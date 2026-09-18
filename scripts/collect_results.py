"""Summarise every results/*.json into one comparison table.

Usage:  python scripts/collect_results.py results/   [--markdown]

Every arm is evaluated with the same code on the same held-out split, so the losses
here are comparable to each other. They are NOT comparable to the paper's published
numbers, which use the authors' own split.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

E2E_STEP = "1/sqrt(n_fast)"


def load(d: Path) -> list[dict]:
    out = []
    for f in sorted(d.glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except json.JSONDecodeError:
            print(f"  ! {f.name}: unreadable, skipped")
            continue
        if "eval" not in r:
            continue
        a = r.get("args", {})
        n_fast = r.get("param_counts", {}).get("fast", 0)
        lr = a.get("inner_lr")
        out.append({
            "file": f.name,
            "arm": r.get("arm"),
            "mode": r.get("mode"),
            "loss": r["eval"]["loss"],
            "n": r["eval"]["num_sequences"],
            "inner": a.get("inner") or "-",
            "inner_lr": lr,
            # Restate the inner LR as a multiple of the e2e-equivalent step, which is the
            # only form that transfers across fast-weight sets of different sizes.
            "xe2e": (lr * math.sqrt(n_fast) if (lr and n_fast) else None),
            "outer_lr": a.get("outer_lr"),
            "rank": a.get("lora_rank"),
            "fast_blocks": a.get("fast_blocks"),
            "steps": a.get("steps") if r.get("mode") == "train" else 0,
            "peak_gib": r.get("peak_gib"),
            "n_fast": n_fast,
            "n_slow": r.get("param_counts", {}).get("slow", 0),
            "final_train_loss": (r.get("history") or [{}])[-1].get("loss"),
            "inner_lr_mult": (r.get("history") or [{}])[-1].get("inner_lr_mult_mean"),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args()
    rows = sorted(load(a.dir), key=lambda r: (str(r["arm"]), r["loss"]))
    if not rows:
        print("no results with an eval block found")
        return

    hdr = ["arm", "file", "loss", "n", "inner", "inner_lr", "xe2e", "outer_lr", "rank",
           "fb", "steps", "peak", "lr_mult"]
    def fmt(r):
        return [str(r["arm"]), r["file"][:26], f"{r['loss']:.4f}", str(r["n"]), str(r["inner"]),
                ("-" if r["inner_lr"] is None else f"{r['inner_lr']:.0e}"),
                ("-" if r["xe2e"] is None else f"{r['xe2e']:.2f}x"),
                ("-" if r["outer_lr"] is None else f"{r['outer_lr']:.0e}"),
                str(r["rank"]), str(r["fast_blocks"]), str(r["steps"]),
                ("-" if r["peak_gib"] is None else f"{r['peak_gib']:.0f}G"),
                ("-" if r["inner_lr_mult"] is None else f"{r['inner_lr_mult']:.3f}")]
    table = [hdr] + [fmt(r) for r in rows]
    w = [max(len(row[i]) for row in table) for i in range(len(hdr))]
    sep = "|" + "|".join("-" * (c + 2) for c in w) + "|" if a.markdown else None
    for i, row in enumerate(table):
        line = "| " + " | ".join(c.ljust(w[j]) for j, c in enumerate(row)) + " |"
        print(line)
        if a.markdown and i == 0:
            print(sep)

    # Compare only against an arm A measured on the SAME number of sequences: a
    # different eval subset is a different dataset, so a cross-n delta is meaningless.
    bases = {r["n"]: r for r in rows if r["arm"] == "A"}
    if bases:
        print("\ndeltas vs arm A on the SAME eval set (negative is better):")
        for r in rows:
            if r["arm"] == "A":
                continue
            b = bases.get(r["n"])
            if b is None:
                print(f"  arm {r['arm']:2s} {r['file'][:30]:32s} n={r['n']:<4d} no arm A at this n; not comparable")
            else:
                print(f"  arm {r['arm']:2s} {r['file'][:30]:32s} n={r['n']:<4d} "
                      f"{r['loss']:.4f} - {b['loss']:.4f} = {r['loss'] - b['loss']:+.4f}")
    print(f"\nxe2e restates the inner LR as a multiple of {E2E_STEP}, the per-element step "
          f"TTT-E2E's clip_by_global_norm(1)+sgd(1) produces for that fast-weight set.")


if __name__ == "__main__":
    main()
