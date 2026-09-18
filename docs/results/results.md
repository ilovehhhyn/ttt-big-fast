# Results

Model: Llama-3.2-1B (`unsloth/Llama-3.2-1B`, ungated mirror, verified identical config
and token ids to `meta-llama/Llama-3.2-1B`). Sliding-window attention k=8192, TTT chunk
b=1024. Data: DCLM-Baseline documents of >=8193 Llama-3 tokens, streamed from
`mlfoundations/dclm-baseline-1.0-parquet` and tokenized locally (199.6M train tokens,
902K val tokens = 110 sequences of 8192). Evaluation is our own held-out split, so these
numbers are internally comparable but NOT comparable to the paper's published table.

Weight-import correctness: our chunked prefix+suffix path reproduces HuggingFace
`transformers` logits on the same input with max abs difference 0.0000 and
correlation 1.000000.

## Arm A - SWA baseline, no TTT (8K context)

| metric | value |
|---|---|
| held-out loss (log ppl) | **2.5516** |
| eval sequences | 64 |
| peak GPU memory | 8.26 GiB |
| eval wall time | 129 s on one A100-80GB |
| fast params (present, never updated) | 201,326,592 |
| slow params | 0 |
| frozen params | 1,034,487,820 |

Loss by token index (the paper's Fig. 6 view):

| token range | mean NLL |
|---|---|
| 0 - 128 | 3.2525 |
| 128 - 512 | 2.6737 |
| 512 - 2048 | 2.5796 |
| 2048 - 8192 | 2.5243 |

The monotone decrease confirms the model is using accumulated context, which is the
behaviour every other arm is measured against.

Arm A runs the same chunked code path as arm C with the inner optimizer set to `none`,
so the A-vs-C comparison isolates test-time training itself rather than any difference
in the compute graph.

## Memory finding (arm C, first attempt)

Arm C initially ran out of memory on an 80 GiB A100. Cause: the model was built in fp32,
and the math SDPA backend that double backward requires materialises a
[heads=32, q=1024, k=9216] attention score matrix per suffix block - 1.12 GiB in fp32,
which is exactly the allocation size the OOM reported. Fixes applied: bf16 autocast on the
forward with fp32 master weights (e2e's compute_dtype/param_dtype split), and an optional
per-block recompute during backward (e2e's `remat_block`). Fast weights stay fp32 because
a unit-norm inner step spread over 2e8 elements moves each element by ~1e-4, which is at
the bf16 resolution of a 0.02-scale weight.

## Measured memory of the second-order path (A100-80GB, Llama-3.2-1B)

Probe: `scripts/memory_probe.py`, bf16 autocast, LoRA r=64, fast_blocks=4.

| stage | peak |
|---|---|
| model resident (fp32 master weights) | 4.66 GiB |
| + prefix forward over the full sequence | 10.90 GiB (8K: 16.50) |
| + one suffix chunk forward | 18.37 GiB |

Sequence length 4096 (4 chunks), varying the checkpoint group size:

| remat_group | forward peak | backward |
|---|---|---|
| 1 | 33.94 GiB | OOM (77.0 GiB) |
| 2 | 65.96 GiB | OOM (77.7 GiB) |
| 4 | OOM in forward | OOM (77.7 GiB) |

Two things this settles:

1. **Checkpointing through time works.** Going from group 2 to group 1 halves forward
   peak (65.96 -> 33.94 GiB), the expected `(N/g + g)` behaviour.
2. **The binding cost is the BACKWARD, not the forward.** At group 1 the forward fits in
   34 GiB, then backward adds ~43 GiB while recomputing a single chunk. The cause is
   double backward through the math SDPA backend, which the second-order path forces
   (fused kernels have no double backward). Forcing MATH materialises a
   [heads, chunk, window+chunk] score matrix and its grad-of-grad intermediates.

So the plan's estimate in section 0.2, which counted only fast-weight copies
`(N/g + g) * |W|`, understated the real requirement: it omitted the attention
double-backward term, which dominates at this model size. The corrected picture is that
per-chunk backward cost scales with `fast_blocks * heads * chunk * (window + chunk)`,
not with the fast-weight count.

One observation worth noting for the record: `torch.autocast` leaves the residual stream
in fp32 (the embedding output is fp32 and each residual add keeps that dtype), so the
prefix output is fp32 even under bf16 autocast. Only the matmuls run in bf16.
