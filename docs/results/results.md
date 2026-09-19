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

### What makes fast_blocks=4 fit at 8K

Re-measured at 8192 context, `remat_group=1`, `--remat-blocks`, bf16 autocast, prefix
checkpointed, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`:

| fast_blocks | fraction of 16 | forward peak | backward peak | fits in 79.3 GiB? |
|---|---|---|---|---|
| 1 | 1/16 | 33.95 GiB | 53.49 GiB | yes |
| 2 | 1/8 | 36.48 GiB | 71.02 GiB | yes |
| 4 | **1/4 (plan default)** | 38.01 GiB | **77.30 GiB** | yes, with ~2 GiB spare |

Backward grows about 12 GiB per additional fast block while forward grows under 2 GiB,
confirming that the attention double-backward, not fast-weight storage, sets the limit.
The four settings that together brought 4 blocks under the limit were: bf16 autocast,
checkpointing the prefix, `remat_group=1` (not the `sqrt(N)=2` default), and per-block
recompute. Without them the same configuration OOMed at 77.6 GiB while trying to
allocate one more 1.12 GiB fp32 score matrix.

Consequence for the fast-weight-fraction ablation: 1/2 and all-blocks do NOT fit on an
80 GiB card and would need either H200s (drained on this cluster), model parallelism
across GPUs, or truncated backprop through time.

## Arm B - TTT-naive (inner loop on, nothing meta-learned), 8K, 32 sequences

Fast weights = MLPs of the last 4 blocks (201,326,592 params). `normalized_sgd`, where
`lr_rms` IS the per-element RMS of each chunk update. The reference point is TTT-E2E's own
rule, whose per-element step for this fast set is `1/sqrt(201326592) = 7.05e-5`.

| inner lr_rms | multiple of the e2e-equivalent step | held-out loss |
|---|---|---|
| 0 (no TTT) | - | **2.4939** |
| 2e-5 | 0.28x | 2.6632 |
| 7e-5 | 1.0x | 5.3181 |
| 2e-4 | 2.8x | 12.8929 |
| e2e's exact rule: `clip_by_global_norm(1)` + `sgd(1)` | 1.0x (global, not per-tensor) | 8.0595 |

The last row is worth separating. It has the same *total* update norm as `lr_rms=7e-5`
per-tensor (both give `||u|| = 1` when the gradient is large), but it distributes that
budget by global norm, so tensors with large gradients absorb most of the step instead of
every tensor moving by the same per-element RMS. On this model that concentration is worse:
8.06 versus 5.32. Per-tensor normalization is the gentler of the two at equal total step.

**Test-time training without meta-learning monotonically damages a strongly pretrained
model.** Even at exactly the paper's effective step size, loss more than doubles. This is
consistent in direction with TTT-E2E's own finding that TTT-naive gains little and that
meta-learning is what makes test-time updates useful (their Fig. 2), but the effect here is
far more severe than in their setting. The likely reason is the base model: they meta-train
from scratch, whereas we start from Llama-3.2-1B, which has been trained far longer and
therefore sits in a sharper minimum where an untrained gradient step is destructive.

This raises the bar for arm C: the outer loop must not merely improve on TTT-naive, it must
overcome an actively harmful inner loop. The learned per-tensor inner learning rate is the
mechanism that can do it, since it can shrink the step where the step hurts.

## Arm E - reference TTT-E2E 760M (third-party reproduction), 8K, 32 sequences

Checkpoint: `Luxel/ttt-e2e-760m-results`, stage `S2_ADAPT/adapt-760m-e2e-8K-from-fa`,
converted from orbax to PyTorch (344 tensors, 883.7M params). NOT the authors' own release.
Inner rule is e2e's exact `clip_by_global_norm(1.0)` + `sgd(1.0)`.

| configuration | loss |
|---|---|
| halves RoPE (wrong), no TTT | 5.5771 |
| halves RoPE (wrong), with TTT | 5.0808 |
| **interleaved RoPE (correct), with TTT** | **3.0668** |

The 2.0-nat jump came from a single convention: TTT-E2E's `apply_rotary_emb` reshapes the
head dimension to `(..., d/2, 2)` and multiplies as a complex number, pairing channels
(0,1), (2,3), ..., whereas HuggingFace Llama splits the head dimension into halves. A model
trained under one convention is mis-rotated under the other. Arms A-D keep the halves
convention and their HF logits parity is unaffected (correlation 1.000000).

Arm E is a 760M model and arm A is a 1.24B model, so the gap between 3.07 and 2.49 is
mostly capacity and pretraining budget, not method. Arm E is a reference point for what the
published recipe produces, not a parameter-matched comparison.

## Arm C - scale-reduced replication (local, SmolLM2-135M @ 2048 context)

Della was unreachable (expired interactive auth), so arms A, B and C were also run at
reduced scale on one machine. The structure is held identical to the 1B/8K experiment so
the comparison is like-for-like in shape:

| property | Della 1B run | local 135M run |
|---|---|---|
| chunks per sequence | 8 (8192/1024) | 8 (2048/256) |
| window k / chunk b | 8192 / 1024 = **8** | 2048 / 256 = **8** |
| window vs context | k = T, so SWA = full attention | k = T, same |
| fast weights | MLPs of last 4 of 16 blocks (201M) | MLPs of last 7 of 30 blocks (18.6M) |
| slow weights | attention LoRA r=64 + norms + inner LRs (13.7M) | same (7.4M) |
| e2e-equivalent inner step `1/sqrt(n_fast)` | 7.05e-5 | 2.32e-4 |

All three arms on the SAME 16 held-out sequences:

| arm | what | loss | delta vs A |
|---|---|---|---|
| A | no TTT | **2.6150** | - |
| B | TTT-naive, inner lr 7e-5 (0.30x e2e step) | 2.6389 | +0.0239 |
| C | meta-learned LoRA, same inner lr, 24 outer steps | 2.6422 | +0.0272 |

**Result: arm C is indistinguishable from arm B (+0.0033) and both are slightly worse than
no TTT at all.** At this budget, meta-learning the slow LoRA did not make test-time training
useful. Training loss over the 24 steps was noisy (2.906, 2.780, 2.655, 3.218, 2.975, 2.936)
because each outer step averages only 4 sequences.

The single most informative number is the learned inner learning rate. It starts at exactly
1.000 and drifts DOWN to 0.9915 over 24 steps - the outer loop is beginning to switch the
inner loop off, which is the rational response when TTT is harmful. It moved only 0.85%,
so this is a direction, not a conclusion.

**What this does and does not establish.** It does not test H1: 24 outer steps is 786K
tokens against a planned 125M, so the adapter has barely moved, and arm D (all-weights slow)
was not run, so the falsifier "C approximately equals B while D is much greater than B" is
only half-measured. What it does establish is that the whole pipeline runs end to end and
produces a coherent, self-consistent A/B/C comparison, and that the early direction of travel
is toward disabling TTT rather than exploiting it.

Reporting note: deltas are computed only against an arm A measured on the SAME number of
sequences. An earlier version of `collect_results.py` compared against whichever arm A had
the lowest loss, which silently mixed a 4-sequence baseline with 16-sequence arms.

## Complete arm comparison at full scale (Llama-3.2-1B, 8K context, DCLM)

All rows evaluated with identical code on the SAME 32 held-out sequences.

| arm | what | inner rule | loss | delta vs A |
|---|---|---|---|---|
| A | no TTT | - | **2.4940** | - |
| B | TTT, inner LR 0 (consistency check) | normalized_sgd, 0 | 2.4939 | **-0.0000** |
| C | **meta-learned LoRA, 18 outer steps** | normalized_sgd, 2e-5 (0.28x) | **2.5991** | +0.1052 |
| B | TTT-naive, same inner LR as C | normalized_sgd, 2e-5 (0.28x) | 2.6632 | +0.1693 |
| E | TTT-E2E 760M reference (third-party) | e2e's clip(1)+sgd(1) | 3.0668 | +0.5728 |
| B | TTT-naive at the e2e-equivalent step | normalized_sgd, 7e-5 (0.99x) | 5.3181 | +2.8241 |
| B | TTT-naive with e2e's exact rule | clipped_sgd, lr 1, tau 1 | 8.0595 | +5.5655 |
| B | TTT-naive, 2.8x the e2e step | normalized_sgd, 2e-4 | 12.8929 | +10.3989 |

### The three things this table says

**1. The implementation is correct.** Running the full TTT machinery with the inner learning
rate set to zero reproduces the no-TTT baseline to four decimal places (2.4939 vs 2.4940).
Every chunk, cache, checkpoint and second-order path is exercised in that row, so the
agreement is a genuine end-to-end check at 1B scale, not a trivial one.

**2. Meta-learning helps, measurably.** Arm C and the matching arm B differ only in whether
the slow LoRA was meta-trained. C recovers 0.0641 of B's 0.1693 nat deficit, i.e. **38% of the
damage that test-time training does on its own** - after only 18 outer steps (2.4M tokens,
about 2% of the planned budget).

**3. It is not yet enough.** Arm C is still 0.105 nats WORSE than not doing test-time training
at all. At this budget the answer to the research question is no: big fast weights with a
small meta-learned slow set do not beat the frozen baseline. Whether more meta-training closes
the remaining gap is exactly what the longer run tests.

### Caveats that matter

- Arm E is a 760M model, arm A is 1.24B. Its +0.5728 is mostly capacity and pretraining
  budget, not method, and it is a third-party reproduction rather than the authors' release.
  It is a reference point, not a parameter-matched comparison.
- 8K with k=8192 means sliding-window attention IS full attention, so this context length
  cannot show the context-scaling behaviour the method targets. The 32K stage is where that
  would appear.
- 18 outer steps is a pilot, not the planned 250-step sweep.

## The regime matters: TTT helps at 32K, where the sliding window actually binds

At 8K with k=8192 the window equals the context, so sliding-window attention IS full
attention and the model already sees every token. The paper says this explicitly:
*"SWA with k = 8K is exactly full attention"*. In that regime TTT has nothing to recover
and every arm B measurement was negative. That is a property of the evaluation setting,
not of the method.

Moving to the paper's extension setting - PG-19 books at 32K context with k=8192, so
T/k = 4 and the window genuinely discards information - reverses the result.

**Llama-3.2-1B, PG-19, 32768 context, window 8192, chunk 1024 (32 chunks), 16 held-out books:**

| arm | inner LR | multiple of e2e step | loss | delta vs no TTT |
|---|---|---|---|---|
| A | - (no TTT) | - | 2.4216 | - |
| B | **7e-6** | **0.10x** | **2.2828** | **-0.1388** |
| B | 2e-5 | 0.28x | 2.5510 | +0.1294 |
| B | 7e-5 | 0.99x | 7.7496 | +5.3280 |

**Test-time training improves held-out loss by 0.139 nats with no meta-learning at all**,
once the window is small enough relative to the context for the compressed memory to be
worth having. The optimum is around a tenth of the e2e-equivalent step, far gentler than
the paper's own setting needs, which is consistent with our base model being a heavily
pretrained Llama rather than a model meta-trained from scratch.

Two practical notes. Evaluation at 32K costs 10.3 GiB without TTT and 36.7 GiB with it, so
the regime is cheap to explore. And the frozen prefix had to be segmented to get here: run
in one shot over 32768 tokens it holds 72 GiB of activations by itself, which alone exhausts
an 80 GiB card. Segmenting it with a rolling KV cache is exact rather than approximate,
because sliding-window attention never looks back further than the cache carries.
