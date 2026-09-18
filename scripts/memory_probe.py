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
    ap.add_argument("--remat-group", type=int, default=0)
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
                                   remat_group=a.remat_group, dtype=a.dtype,
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

    torch.cuda.reset_peak_memory_stats()
    with loop._autocast("cuda"):
        prefix = model.prefix_forward(ids)
    print(f"prefix dtype={prefix.dtype} peak={gib(torch.cuda.max_memory_allocated()):.2f} GiB", flush=True)

    torch.cuda.reset_peak_memory_stats()
    caches = model.init_caches(batch=1, device=dev, dtype=prefix.dtype)
    with loop._autocast("cuda"):
        logits, _ = model.suffix_forward(prefix[:, :a.chunk], fast=dict(split.fast),
                                         caches=caches, chunk_index=0)
    print(f"one suffix chunk: logits dtype={logits.dtype} attn_out={seen.get('attn_out_dtype')} "
          f"peak={gib(torch.cuda.max_memory_allocated()):.2f} GiB", flush=True)

    torch.cuda.reset_peak_memory_stats()
    try:
        out = loop.run_sequence(ids, tgt, mask, dict(split.fast), lr_scale=1.0,
                                lr_mult=model.inner_lr_multipliers())
        print(f"full sequence fwd: loss={out.loss.item():.4f} "
              f"peak={gib(torch.cuda.max_memory_allocated()):.2f} GiB", flush=True)
        out.loss.backward()
        print(f"after backward:    peak={gib(torch.cuda.max_memory_allocated()):.2f} GiB", flush=True)
    except torch.OutOfMemoryError as e:
        print(f"OOM: {str(e)[:160]}", flush=True)
        print(f"peak at OOM={gib(torch.cuda.max_memory_allocated()):.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
