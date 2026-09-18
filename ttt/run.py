"""Experiment runner.

Arms (plan section 0.1). All share the same model, data and evaluation code; they
differ only in which parameters are fast, which are slow, and whether the inner
loop takes a step.

    A  baseline   fast MLPs present but inner optimizer = none  -> no TTT at all.
                  Using the same chunked path as arm C (rather than a plain forward)
                  keeps the compute graph identical, so A vs C isolates TTT itself.
    B  ttt-naive  inner loop on, nothing slow -> dynamic evaluation, no meta-learning.
    C  proposed   fast = MLPs of the last `fast_blocks`; slow = attention LoRA +
                  RMSNorm gains + learned per-tensor inner LRs.
    D  full-slow  same fast set, every parameter slow.
    F  small-fast paper-style: fewer fast blocks, same slow set as C.

Usage:
    python -m ttt.run --arm A --mode eval  --data DIR --out results/A.json
    python -m ttt.run --arm C --mode train --data DIR --out results/C.json --steps 250
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from ttt.config import Config, InnerConfig, LoRAConfig, OuterConfig, TrainConfig
from ttt.data.dataset import build_dataloader
from ttt.eval.evaluator import evaluate
from ttt.model.naming import split_parameters
from ttt.optim.inner import build_inner_optimizer
from ttt.optim.outer import build_outer_optimizer
from ttt.train.inner_loop import TTTInnerLoop
from ttt.train.trainer import Trainer
from ttt.utils.hf_import import MIRROR_REPO, build_llama_ttt

ARMS = {
    "A": dict(inner="none", lora_rank=0, slow=()),
    "B": dict(inner="normalized_sgd", lora_rank=0, slow=()),
    "C": dict(inner="normalized_sgd", lora_rank=64, slow=("lora_A", "lora_B", "norm.weight", "inner_lr_log")),
    "D": dict(inner="normalized_sgd", lora_rank=0, slow=("**",)),
    "F": dict(inner="normalized_sgd", lora_rank=64, slow=("lora_A", "lora_B", "norm.weight", "inner_lr_log")),
}


def build_everything(args) -> tuple[Config, torch.nn.Module, object, TTTInnerLoop, torch.device]:
    arm = ARMS[args.arm]
    device = torch.device(args.device)
    lora = LoRAConfig(rank=args.lora_rank if args.lora_rank is not None else arm["lora_rank"],
                      alpha=args.lora_alpha, scaling="rslora",
                      targets=tuple(args.lora_targets.split(",")))
    model = build_llama_ttt(args.repo, max_seq_len=args.seq_len, window_size=args.window,
                            chunk_size=args.chunk, fast_blocks=args.fast_blocks,
                            lora=lora if lora.rank > 0 else None,
                            dtype=torch.float32, cache_dir=args.hf_cache)
    # Master weights stay fp32; the forward runs under bf16 autocast (see
    # TTTInnerLoop._autocast). remat_blocks trades compute for the math-SDPA score
    # matrices, which dominate activation memory in the second-order path.
    model.remat_blocks = args.remat_blocks
    model = model.to(device)

    inner = InnerConfig(optimizer=arm["inner"] if args.inner is None else args.inner,
                        lr_rms=args.inner_lr, norm_scope=args.norm_scope,
                        eps=args.adam_eps, clip_tau=args.clip_tau, beta1=0.9, beta2=0.9, warm_start=True,
                        learned_lr=bool(arm["slow"]) and "inner_lr_log" in arm["slow"],
                        delta_decay=args.delta_decay)
    outer = OuterConfig(lr=args.outer_lr, total_steps=args.steps)
    train = TrainConfig(seq_len=args.seq_len, tokens_per_step=args.tokens_per_step,
                        micro_batch=1, remat_group=args.remat_group,
                        slow_spec=arm["slow"] or ("__none__",), dtype=args.dtype)
    cfg = Config(model=model.cfg, inner=inner, outer=outer, train=train)
    split = split_parameters(model, model.cfg, cfg.train)
    loop = TTTInnerLoop(model, cfg, build_inner_optimizer(cfg.inner))
    return cfg, model, split, loop, device


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=sorted(ARMS))
    p.add_argument("--mode", required=True, choices=["eval", "train"])
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--repo", default=MIRROR_REPO)
    p.add_argument("--hf-cache", default=None)
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--window", type=int, default=8192)
    p.add_argument("--fast-blocks", type=int, default=4)
    p.add_argument("--remat-group", type=int, default=0)
    p.add_argument("--remat-blocks", action="store_true",
                   help="recompute each suffix block during backward (big memory win)")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--tokens-per-step", type=int, default=524288)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--inner", default=None,
                   choices=[None, "none", "normalized_sgd", "adamw", "muon", "clipped_sgd"])
    p.add_argument("--inner-lr", type=float, default=1e-3)
    p.add_argument("--norm-scope", default="tensor", choices=["tensor", "global"])
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--clip-tau", type=float, default=1.0)
    p.add_argument("--delta-decay", type=float, default=0.0)
    p.add_argument("--outer-lr", type=float, default=1e-3)
    p.add_argument("--lora-rank", type=int, default=None)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-targets", default="wq,wk,wv,wo")
    p.add_argument("--eval-sequences", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    cfg, model, split, loop, device = build_everything(args)
    counts = {k: sum(v.numel() for v in getattr(split, k).values()) for k in ("fast", "slow", "frozen")}
    print(f"[run] arm={args.arm} fast={counts['fast']:,} slow={counts['slow']:,} frozen={counts['frozen']:,}", flush=True)
    print(f"[run] chunks={cfg.num_chunks} remat_group={loop.group} seqs_per_step={cfg.train.seqs_per_step}", flush=True)

    result = {"arm": args.arm, "mode": args.mode, "args": vars(args), "param_counts": counts,
              "num_chunks": cfg.num_chunks, "remat_group": loop.group}

    if args.mode == "train":
        opt = build_outer_optimizer(split.slow, cfg.outer)
        train_loader = build_dataloader(Path(args.data), "train", args.seq_len, 1,
                                        shuffle=True, seed=args.seed, num_workers=2)
        it = iter(_cycle(train_loader))
        trainer = Trainer(cfg, model, split, loop, opt, it, device=device)
        history = []
        for step in range(args.steps):
            m = trainer.train_step(step)
            history.append(m.as_log())
            if step % 5 == 0 or step == args.steps - 1:
                print(f"[train] {m.as_log()}", flush=True)
        result["history"] = history

    val_loader = build_dataloader(Path(args.data), "val", args.seq_len, 1,
                                  shuffle=False, seed=args.seed, num_workers=2)
    t0 = time.perf_counter()
    ev = evaluate(loop, split, val_loader, max_sequences=args.eval_sequences, device=device)
    result["eval"] = {"loss": ev.loss, "num_sequences": ev.num_sequences,
                      "token_nll": ev.token_nll.tolist(), "seconds": time.perf_counter() - t0}
    if torch.cuda.is_available():
        result["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
    print(f"[eval] arm={args.arm} loss={ev.loss:.4f} n={ev.num_sequences} "
          f"peak={result.get('peak_gib', 0):.1f}GiB", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[run] wrote {args.out}", flush=True)


def _cycle(loader):
    while True:
        yield from loader


if __name__ == "__main__":
    main()
