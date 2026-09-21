"""Recall from beyond the attention window, for ONE model configuration (ttt/eval/recall.py).

Takes every option of `python -m ttt.run` (the model is built by the same code, and
--load-slow evaluates trained slow weights), plus where to plant the passage:

    python scripts/recall_probe.py --arm C --mode eval --data DIR --seq-len 32768 --chunk 1024 \\
        --window 1024 --prefix-segment 1024 --fast-blocks 4 --remat-group 1 --remat-blocks \\
        --inner normalized_sgd --inner-lr 4e-6 --eval-ttt-off --load-slow results/X.ckpt \\
        --gap 17408 --out results/recall_X.json

    recall = NLL_absent - NLL_present on the repeated passage   (nats per token, positive = recalled)

With --eval-ttt-off the SAME weights are also probed with the inner loop off. Whenever the gap
exceeds the receptive field L*(k-1) of the windowed attention, a run without test-time training
must give recall == 0 exactly; the script checks that and exits non-zero if it does not hold,
because then the two inputs differ in something other than what the model may remember.
The ceiling is the same command with full attention: --arm A --window <seq-len>.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from ttt.config import Config
from ttt.data.dataset import TokenSequenceDataset, _shard_indices
from ttt.eval.forgetting import lr_multipliers
from ttt.eval.paired import clustered_paired_stats
from ttt.eval.recall import RecallSpec, choose_pairs, score_pair
from ttt.optim.inner import build_inner_optimizer
from ttt.run import build_everything, build_parser
from ttt.train.checkpoint import load_slow_weights
from ttt.train.inner_loop import TTTInnerLoop

EXACT_FLOOR_TOL = 1e-6  # nats; identical computations should agree to the last bit


def _line(label: str, s: dict) -> str:
    return (f"{label:<34} {s['mean']:+.4f}  95% CI [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]  "
            f"positive {s['positive']}/{s['n']} documents")


def main() -> None:
    p = build_parser()
    p.add_argument("--source-start", type=int, default=2048, help="a: the passage is planted at tokens [a, a+n)")
    p.add_argument("--gap", type=int, required=True,
                   help="g: tokens between the end of the planted passage and its repeat at b = a + n + g")
    p.add_argument("--length", type=int, default=1024,
                   help="n: passage length. The default is one chunk, planted chunk-aligned, so that ONE inner "
                        "step is taken on exactly the passage")
    p.add_argument("--cue", type=int, default=32, help="first tokens of the repeat that are context, not scored")
    p.add_argument("--pairs", type=int, default=32, help="number of (carrier, donor) pairs; --eval-sequences is not used")
    args = p.parse_args()
    assert args.mode == "eval", "the recall probe is evaluation only: pass --mode eval"
    spec = RecallSpec(source_start=args.source_start, gap=args.gap, length=args.length, cue=args.cue)
    spec.check(args.seq_len)

    torch.manual_seed(args.seed)
    cfg, model, split, loop, device = build_everything(args)
    result = {"args": vars(args), "spec": {**asdict(spec), "target_start": spec.target_start}}
    if args.load_slow:
        result["loaded_slow"] = load_slow_weights(Path(args.load_slow), split=split)
        print(f"[load-slow] {args.load_slow}: step {result['loaded_slow']['step']}", flush=True)

    # Which conditions run. Without an inner optimizer there is only one, and it IS "no TTT".
    has_ttt = cfg.inner.optimizer != "none"
    loops = {"ttt_on" if has_ttt else "no_ttt": loop}
    if has_ttt and args.eval_ttt_off:
        off_cfg = Config(model=cfg.model, inner=replace(cfg.inner, lr_rms=0.0), outer=cfg.outer, train=cfg.train)
        loops["ttt_off"] = TTTInnerLoop(model, off_cfg, build_inner_optimizer(off_cfg.inner))

    L, k = cfg.model.num_layers, cfg.model.window_size
    exact_floor = spec.beyond_receptive_field(L, k)
    result["receptive_field"] = {"layers": L, "window": k, "reach": L * (k - 1), "gap_beyond_reach": exact_floor}
    print(f"[probe] plant [{spec.source_start}, {spec.source_start + spec.length})  repeat at {spec.target_start}  "
          f"gap {spec.gap}  scored tokens {spec.length - spec.cue}  |  attention reach L*(k-1) = {L * (k - 1)}: "
          f"{'the repeat is OUT of reach, no-TTT recall must be exactly 0' if exact_floor else 'the repeat is within reach'}",
          flush=True)

    ds = TokenSequenceDataset(Path(args.data), "val", args.seq_len)
    order = _shard_indices(len(ds), shuffle=True, seed=args.seed, rank=0, world_size=1)
    pairs, skipped = choose_pairs(np.asarray(ds.tokens), ds.bos_token_id, args.seq_len, order, spec, args.pairs)
    result["pairs"] = [asdict(q) for q in pairs]
    result["carriers_skipped_for_document_boundary"] = skipped
    docs = [q.carrier_doc for q in pairs]
    print(f"[probe] {len(pairs)} pairs, carriers from {len(set(docs))} documents "
          f"({skipped} candidates skipped: document boundary between plant and repeat)", flush=True)

    a, n, T = spec.source_start, spec.length, args.seq_len
    nll = {name: {"present": [], "absent": []} for name in loops}
    t0 = time.perf_counter()
    for i, q in enumerate(pairs):
        window = torch.from_numpy(np.asarray(ds.tokens[q.carrier * T : q.carrier * T + T + 1], dtype=np.int64))
        passage = torch.from_numpy(np.asarray(ds.tokens[q.donor * T + a : q.donor * T + a + n], dtype=np.int64))
        for name, lp in loops.items():
            scored = score_pair(lp, split, lr_multipliers(lp), window, passage, spec, device=device)
            for cond in ("present", "absent"):
                nll[name][cond].append(scored[cond])
        msg = "  ".join(f"{name}: absent {nll[name]['absent'][-1].mean():.4f} present {nll[name]['present'][-1].mean():.4f}"
                        for name in loops)
        print(f"[{i + 1}/{len(pairs)}] {msg}", flush=True)
    result["seconds"] = time.perf_counter() - t0

    floor_violations = []
    result["conditions"] = {}
    print(f"\nrecall = NLL_absent - NLL_present over {n - spec.cue} scored tokens, per carrier document")
    for name in loops:
        present, absent = np.stack(nll[name]["present"]), np.stack(nll[name]["absent"])  # [pairs, n - cue]
        recall = (absent - present).mean(axis=1)  # [pairs]
        entry = {"nll_present": float(present.mean()), "nll_absent": float(absent.mean()),
                 "per_pair_recall": recall.tolist(), "per_document": clustered_paired_stats(recall.tolist(), docs),
                 "recall_by_offset": (absent - present).mean(axis=0).tolist(),  # [n - cue], offset 0 = token P[cue]
                 "max_abs_token_difference": float(np.abs(absent - present).max())}
        result["conditions"][name] = entry
        print(_line(f"{name}: absent {entry['nll_absent']:.4f} present {entry['nll_present']:.4f}", entry["per_document"]))
        if name != "ttt_on" and exact_floor and entry["max_abs_token_difference"] > EXACT_FLOOR_TOL:
            floor_violations.append((name, entry["max_abs_token_difference"]))
    if "ttt_on" in loops and "ttt_off" in loops:
        diff = (np.asarray(result["conditions"]["ttt_on"]["per_pair_recall"])
                - np.asarray(result["conditions"]["ttt_off"]["per_pair_recall"]))
        result["ttt_on_minus_off"] = clustered_paired_stats(diff.tolist(), docs)
        print(_line("what TTT adds (on - off)", result["ttt_on_minus_off"]))
    result["exact_floor_checked"] = exact_floor and any(name != "ttt_on" for name in loops)
    result["exact_floor_violations"] = floor_violations

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(args.out).with_name(Path(args.out).name + ".tmp")
    tmp.write_text(json.dumps(result))
    os.replace(tmp, args.out)
    print(f"wrote {args.out}")
    if result["exact_floor_checked"] and not floor_violations:
        print("floor check passed: without test-time training the two conditions agree exactly")
    if floor_violations:
        print(f"FLOOR CHECK FAILED: {floor_violations}. Without TTT and with the repeat out of attention's reach the "
              f"two conditions must agree exactly; they do not, so these recall numbers are not to be trusted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
