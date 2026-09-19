"""Loss broken down by token index, reproducing TTT-E2E Figure 6.

For each token position t the reported value is the loss of predicting x_t given
x_0..x_{t-1}. For a TTT arm this is also the test-time training loss l_t(W_{t-1});
the paper's footnote 8 notes the comparison against non-TTT methods is fair because
W_{t-1} has never seen x_t. Our evaluator computes exactly this loss-before-update.

Their finding is that TTT's aggregate advantage comes mostly from EARLIER tokens.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--pattern", default="pg19_32k")
    ap.add_argument("--chunk", type=int, default=1024)
    a = ap.parse_args()

    curves: dict[str, np.ndarray] = {}
    for f in sorted(a.results_dir.glob("*.json")):
        if a.pattern not in f.name:
            continue
        d = json.loads(f.read_text())
        nll = d.get("eval", {}).get("token_nll")
        if not nll:
            continue
        lr = d.get("args", {}).get("inner_lr")
        inner = d.get("args", {}).get("inner") or "none"
        label = f"{d['arm']} {inner if inner!='none' else 'noTTT'}" + (f" {lr:.0e}" if inner != "none" else "")
        curves[label] = np.asarray(nll, dtype=float)
    assert curves, f"no results matching {a.pattern!r} with a token_nll curve"

    base_key = next((k for k in curves if "noTTT" in k), None)
    T = len(next(iter(curves.values())))
    edges = [0, a.chunk, 2 * a.chunk, 4 * a.chunk, 8 * a.chunk, 16 * a.chunk, T]
    edges = sorted({min(e, T) for e in edges})

    names = list(curves)
    print(f"{'token range':>16} | " + " | ".join(f"{n:>16}" for n in names))
    print("-" * 16 + "-+-" + "-+-".join("-" * 16 for _ in names))
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        cells = " | ".join(f"{curves[n][lo:hi].mean():16.4f}" for n in names)
        print(f"{lo:6d}-{hi:<9d} | {cells}")

    if base_key:
        print(f"\ndelta vs {base_key} (negative = TTT better):")
        base = curves[base_key]
        for n, v in curves.items():
            if n == base_key:
                continue
            parts = [f"{lo//1024}K-{hi//1024}K:{v[lo:hi].mean() - base[lo:hi].mean():+.4f}"
                     for lo, hi in zip(edges[:-1], edges[1:], strict=False)]
            print(f"  {n:>16}  " + "  ".join(parts))
            print(f"  {'':>16}  overall {v.mean() - base.mean():+.4f}")


if __name__ == "__main__":
    main()
