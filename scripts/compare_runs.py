"""Compare two ttt.run result files step by step.

Used to check that a killed-and-resumed run reproduces an uninterrupted run of the same
configuration. On CPU the two are bit-identical; on GPU some kernels are nondeterministic,
so agreement is reported rather than asserted and the caller supplies the tolerance.

    python scripts/compare_runs.py reference.json candidate.json --tol 1e-3
"""
from __future__ import annotations

import argparse
import json
import sys

# Fields that legitimately differ between two runs of the same experiment.
WALL_CLOCK = {"sec_per_step"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="largest acceptable |reference - candidate| on any training metric or eval loss")
    a = ap.parse_args()

    ref, cand = json.load(open(a.reference)), json.load(open(a.candidate))
    print(f"reference resumed_from_step={ref.get('resumed_from_step', 0)}  "
          f"candidate resumed_from_step={cand.get('resumed_from_step', 0)}")

    hr, hc = ref["history"], cand["history"]
    assert len(hr) == len(hc), f"step counts differ: {len(hr)} vs {len(hc)}"
    worst, worst_at = 0.0, None
    print(f"{'step':>4} {'ref loss':>12} {'cand loss':>12} {'|diff|':>10}")
    for er, ec in zip(hr, hc, strict=True):
        assert er["step"] == ec["step"], (er["step"], ec["step"])
        for k in er.keys() - WALL_CLOCK:
            d = abs(er[k] - ec[k])
            if d > worst:
                worst, worst_at = d, (er["step"], k)
        print(f"{er['step']:>4} {er['loss']:>12.6f} {ec['loss']:>12.6f} {abs(er['loss'] - ec['loss']):>10.2e}")
    print(f"largest training-metric difference: {worst:.3e} at (step, metric) = {worst_at}")

    ok = worst <= a.tol
    for key in ("eval", "eval_ttt_off"):
        if key in ref and key in cand:
            d = abs(ref[key]["loss"] - cand[key]["loss"])
            ok = ok and d <= a.tol
            print(f"{key:13s} ref={ref[key]['loss']:.6f} cand={cand[key]['loss']:.6f} |diff|={d:.2e}")
    print("AGREE within tolerance" if ok else f"DISAGREE beyond tolerance {a.tol}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
