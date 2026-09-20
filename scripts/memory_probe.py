"""Diagnose where the second-order path spends memory, one component at a time."""
from __future__ import annotations

import argparse

import torch

from ttt.config import Config, InnerConfig, LoRAConfig, OuterConfig, TrainConfig
from ttt.model.naming import split_parameters
from ttt.optim.inner import build_inner_optimizer
from ttt.train.inner_loop import TTTInnerLoop
from ttt.utils.hf_import import MIRROR_REPO, build_llama_ttt


def gib(x): return x / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--fast-blocks", type=int, default=4)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--remat-blocks", action="store_true")
    ap.add_argument("--prefix-segment", type=int, default=0)
    ap.add_argument("--remat-group", type=int, default=0)
    ap.add_argument("--truncate-bptt", type=int, default=0)
    ap.add_argument("--inference", action="store_true",
                    help="measure the EVAL path (no meta-gradient) instead of training")
    ap.add_argument("--staged", action="store_true",
                    help="also run the grad-enabled staged measurements (they retain their graph)")
    a = ap.parse_args()

    dev = torch.device("cuda")
    model = build_llama_ttt(MIRROR_REPO, max_seq_len=a.seq_len, window_size=8192,
                            chunk_size=a.chunk, fast_blocks=a.fast_blocks,
                            lora=LoRAConfig(rank=64, alpha=16.0), dtype=torch.float32).to(dev)
    model.remat_blocks = a.remat_blocks
    cfg = Config(model=model.cfg,
                 inner=InnerConfig(optimizer="normalized_sgd", lr_rms=1e-3, learned_lr=True),
                 outer=OuterConfig(lr=1e-3, total_steps=10),
                 train=TrainConfig(seq_len=a.seq_len, tokens_per_step=a.seq_len,
                                   remat_group=a.remat_group, dtype=a.dtype, prefix_segment=a.prefix_segment,
                                   truncate_bptt=a.truncate_bptt,
                                   slow_spec=("lora_A", "lora_B", "norm.weight", "inner_lr_log")))
    split = split_parameters(model, model.cfg, cfg.train)
    loop = TTTInnerLoop(model, cfg, build_inner_optimizer(cfg.inner))
    print(f"config: dtype={a.dtype} fast_blocks={a.fast_blocks} chunk={a.chunk} "
          f"chunks={cfg.num_chunks} group={loop.group} remat_blocks={a.remat_blocks}", flush=True)
    print(f"after load: {gib(torch.cuda.memory_allocated()):.2f} GiB", flush=True)

    ids = torch.randint(0, model.cfg.vocab_size, (1, a.seq_len), device=dev)
    tgt = torch.randint(0, model.cfg.vocab_size, (1, a.seq_len), device=dev)
    mask = torch.ones_like(tgt, dtype=torch.float32)

    # Record the dtype the attention actually runs in.
    seen = {}
    def hook(mod, inp, out):
        seen.setdefault("attn_out_dtype", out[0].dtype)
    model.blocks[-1].attn.register_forward_hook(hook)

    # Measured FIRST, on a clean allocator: run_sequence wraps the prefix exactly like
    # this, so this is the real floor every chunk then builds on. The staged grad-enabled
    # calls below retain their graph, so anything measured after them is inflated.
    import gc
    torch.cuda.reset_peak_memory_stats()
    def _prefix_ckpt(i):
        with loop._autocast("cuda"):
            return model.prefix_forward(i, segment=(a.prefix_segment or None))
    pc = torch.utils.checkpoint.checkpoint(_prefix_ckpt, ids, use_reentrant=False)
    print(f"prefix (checkpointed): resident={gib(torch.cuda.memory_allocated()):.2f} GiB "
          f"peak={gib(torch.cuda.max_memory_allocated()):.2f} GiB", flush=True)
    del pc; gc.collect(); torch.cuda.empty_cache()
    print(f"after freeing prefix:  resident={gib(torch.cuda.memory_allocated()):.2f} GiB", flush=True)

    if a.staged:
        torch.cuda.reset_peak_memory_stats()
        with loop._autocast("cuda"):
            prefix = model.prefix_forward(ids, segment=(a.prefix_segment or None))
        print(f"prefix (raw) dtype={prefix.dtype} peak={gib(torch.cuda.max_memory_allocated()):.2f} GiB", flush=True)

        torch.cuda.reset_peak_memory_stats()
        caches = model.init_caches(batch=1, device=dev, dtype=prefix.dtype)
        with loop._autocast("cuda"):
            logits, _ = model.suffix_forward(prefix[:, :a.chunk], fast=dict(split.fast),
                                             caches=caches, chunk_index=0)
        print(f"one suffix chunk: logits dtype={logits.dtype} attn_out={seen.get('attn_out_dtype')} "
              f"peak={gib(torch.cuda.max_memory_allocated()):.2f} GiB", flush=True)
        del prefix, logits, caches
        gc.collect(); torch.cuda.empty_cache()
        print(f"after staged:          resident={gib(torch.cuda.memory_allocated()):.2f} GiB", flush=True)

    # Per-group growth: allocated memory after each checkpointed group, plus the size
    # of the carry itself. The slope separates "the carry is big" from "something else
    # accumulates per chunk", which the fast_blocks 4-vs-1 comparison could not.
    def on_group(start, flat):
        carry = sum(x.numel() * x.element_size() for x in flat)
        print(f"  group@{start:3d}: allocated={gib(torch.cuda.memory_allocated()):6.2f} GiB "
              f"carry={gib(carry):5.2f} GiB", flush=True)
    loop.on_group = on_group

    torch.cuda.reset_peak_memory_stats()
    try:
        out = loop.run_sequence(
            ids, tgt, mask, dict(split.fast), lr_scale=1.0,
            lr_mult=model.inner_lr_multipliers(),
            backward_scale=None if a.inference else 1.0, inference=a.inference)
        print(f"full sequence fwd: loss={out.loss.item():.4f} "
              f"peak={gib(torch.cuda.max_memory_allocated()):.2f} GiB "
              f"backward_done={out.backward_done}", flush=True)
        if not out.backward_done:
            out.loss.backward()
        print(f"after backward:    peak={gib(torch.cuda.max_memory_allocated()):.2f} GiB", flush=True)
    except torch.OutOfMemoryError as e:
        print(f"OOM: {str(e)[:160]}", flush=True)
        print(f"peak at OOM={gib(torch.cuda.max_memory_allocated()):.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
