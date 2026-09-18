# PyTorch Architecture Addendum (supersedes the JAX tasks in 2026-09-17-big-fast-small-slow-ttt.md)

All **fixed values** in §0.2 of the 2026-09-17 plan (model, data, chunk size, optimizer formulas, LoRA, batch size, budgets, seeds, run order A→C→E→B→D→F) are framework-independent and carry over **unchanged**. This document replaces only the architecture and Tasks 1–10.

## Why PyTorch changes the design (three hard constraints)

1. **No `torch.compile` anywhere in the meta-gradient region.** `create_graph=True` inside a compiled region is unsupported (pytorch#91469). Only the frozen prefix blocks may be compiled.
2. **No fused attention in blocks at or above the first fast block.** FlashAttention and mem-efficient SDPA have no double backward. Those blocks use `SDPBackend.MATH`. Blocks strictly below the first fast block have no second-order path (their forward does not depend on any fast weight), so they keep the fused kernel. This is exactly e2e's prefix/suffix split.
3. **Fast weights must be functional tensors, not `nn.Parameter` leaves.** An in-place optimizer step would break the graph. The inner loop threads a `dict[str, Tensor]` through `torch.func.functional_call`.

## Module layout

```
ttt/
  config.py                 # frozen dataclasses, Hydra structured configs
  model/
    rope.py                 # llama3 scaled RoPE
    lora.py                 # LoRALinear
    mlp.py                  # SwiGLUMLP (fast-weight carrier)
    attention.py            # SlidingWindowAttention (GQA, KV cache, backend switch)
    block.py                # TransformerBlock
    transformer.py          # TTTTransformer: prefix_forward / suffix_forward / lm_head
    naming.py               # fast/slow parameter selection by name pattern
  optim/
    inner.py                # InnerOptimizer ABC + NormalizedSGD / DifferentiableAdamW / MuonNoMomentum
    outer.py                # AdamW + warmup-cosine schedule, param groups
  train/
    inner_loop.py           # chunked TTT with checkpoint-through-time  <-- core
    trainer.py              # outer loop, grad accumulation, DDP
  data/
    prepare.py              # DCLM parquet / PG-19 -> uint32 memmap shards
    dataset.py              # fixed-length sequence sampler
  eval/
    evaluator.py            # held-out loss + per-token-index curve
    forgetting.py           # forgetting probe
  utils/
    hf_import.py            # Llama-3.2-1B safetensors -> TTTTransformer
    dist.py                 # DDP init helpers
tests/                      # pytest, CPU-only, tiny configs
configs/                    # Hydra YAML, one per arm
scripts/della/              # sbatch templates
```

## Core mechanism: the inner loop (`train/inner_loop.py`)

```
fast   : dict[str, Tensor]     # W_i, differentiable, starts as a view of W0
slow   : dict[str, Tensor]     # LoRA A/B, norm gains, inner-LR logits (nn.Parameters)
frozen : dict[str, Tensor]     # everything else, requires_grad=False

for group in chunks.split(n):                      # n = round(sqrt(N))
    fast, cache = checkpoint(run_group, fast, cache, group, use_reentrant=False)

run_group(fast, cache, group):
    for chunk in group:
        logits, cache = suffix_forward(fast|slow|frozen, prefix_out[chunk], cache)
        loss_i  = cross_entropy(logits, targets[chunk])       # loss BEFORE the update
        g       = autograd.grad(loss_i, fast.values(), create_graph=True)
        fast    = inner_opt.step(fast, g, state)              # W_i = W_{i-1} - update
    return fast, cache

outer_loss = mean_i(loss_i)                         # Eq. 6: loss before update, averaged
```

Invariants that must hold and are unit-tested:
- `loss_i` is computed with `W_{i-1}`, never `W_i`.
- Chunk count `N = T // b` exactly; `T % b == 0` asserted.
- The KV cache holds exactly `min(window, tokens_so_far)` positions and is part of the checkpointed carry, so gradients reach fast weights of earlier chunks.
- With `inner_lr = 0` the outer gradient equals the plain (no-TTT) gradient. Tested.
- Fast weights are reset to W0 at every sequence boundary.

## Task list (PyTorch)

Ordered for the A→C→E→B→D→F run order; Tasks 1–8 are shared infrastructure.

| Task | Deliverable | Parallelizable |
|---|---|---|
| 1 | Repo skeleton, config dataclasses, CI-able pytest | — |
| 2 | `rope.py`, `mlp.py`, `attention.py`, `block.py` + parity tests vs HF | yes (agent) |
| 3 | `lora.py` + `naming.py` (fast/slow selection) | yes (agent) |
| 4 | `optim/inner.py` (3 optimizers) + gradient-flow tests | yes (agent) |
| 5 | `data/prepare.py` + `dataset.py` | yes (agent) |
| 6 | `transformer.py` + `hf_import.py`, logits parity vs HF | after 2,3 |
| 7 | `train/inner_loop.py` + invariant tests | after 4,6 |
| 8 | `train/trainer.py`, `optim/outer.py`, `eval/*`, configs, sbatch | after 7 |
| 9 | Memory probe on 1 H100; tune `inner_remat_freq` | after 8 |
| 10 | Sweeps then headline runs in order A, C, E, B, D, F | after 9 |

## Testing rules (TDD)
- Every module gets its test written first, run to confirm failure, then implemented.
- All unit tests run on CPU with a tiny config (2 layers, d=32, 4 heads, 2 KV heads, ff=64, vocab=128, window=8, chunk=4, T=16) in under 10 s total.
- Numerical tests use `torch.float64` where exactness matters (gradient checks via `torch.autograd.gradcheck`).
- No `try/except` fallbacks. Invalid config raises immediately.
