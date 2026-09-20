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

Units are comparable with arms A-D. Every model config in the reference repository sets
`vocab_size: 128256` and its README describes the datasets as Llama-3 tokenized, so arm E
uses the same tokenizer as Llama-3.2-1B and its nats per token measure the same thing. (The
`Llama-2-7b-hf` tokenizer and 32000 vocabulary in their `ttt/config.py` are dataclass
defaults that the experiment YAMLs override.) Their loss is `jax.nn.log_softmax`, natural
log, with logits upcast to float32 first, matching `masked_cross_entropy` here.

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

**What this does and does not establish.** It does not test H1: 24 outer steps at 8192 tokens per step is 196,608 (not 786K, as this line once said)
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
| B | 2e-6 | 0.028x | 2.3062 | -0.1154 |
| B | **4e-6** | **0.057x** | **2.2669** | **-0.1547** |
| B | 7e-6 | 0.099x | 2.2828 | -0.1388 |
| B | 1e-5 | 0.14x | 2.3296 | -0.0920 |
| B | 2e-5 | 0.28x | 2.5510 | +0.1294 |
| B | 7e-5 | 0.99x | 7.7496 | +5.3280 |

The curve is cleanly U-shaped with an interior optimum at 0.057x the e2e-equivalent step.
Every setting from 0.028x to 0.14x beats no-TTT; the method is not knife-edge sensitive,
it simply needs a step roughly an order of magnitude gentler than the paper's own.

(Superseded: the figures in this paragraph predate the contamination fix below; the corrected values are -0.1424 nats at an optimum of 0.057x the e2e-equivalent step.)
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

### Corrected 32K numbers, and an evaluation flaw worth recording

The first 32K evaluation was contaminated. PG-19's validation split opens with the King
James Bible, which the base model has memorised: 0.19 nats and only 1,039 distinct tokens
over 7K positions. That book is long enough that 16 sequences of 32768 tokens never left
it, so the entire evaluation sat on trivially predictable text. All arms saw the same data,
so the deltas were directionally right, but the absolute numbers were meaningless.

Fixed by shuffling the validation split with a fixed seed: still deterministic and
reproducible, but the sequences now spread across books.

**Llama-3.2-1B, PG-19, 32768 context, window 8192, 32 held-out sequences across books:**

| arm | inner LR | x e2e step | loss | delta vs no TTT |
|---|---|---|---|---|
| A | - | - | 3.7119 | - |
| B | 2e-6 | 0.028x | 3.5995 | -0.1124 |
| B | **4e-6** | **0.057x** | **3.5694** | **-0.1424** |
| B | 7e-6 | 0.099x | 3.5862 | -0.1257 |
| B | 2e-5 | 0.28x | 3.7788 | +0.0670 |

The result survives the fix: same optimum, same U-shape, -0.142 nats.

### Where the gain comes from (paper Fig. 6 analysis, on clean data)

Loss by token index at the optimal inner LR, against no TTT:

| token range | arm A loss | delta with TTT |
|---|---|---|
| 0 - 1K | 2.5301 | +0.0000 |
| 1K - 2K | 2.3600 | +0.0002 |
| 2K - 4K | 2.3261 | +0.0010 |
| 4K - 8K | 2.2701 | +0.0021 |
| 8K - 16K | 4.2054 | **-0.1132** |
| 16K - 32K | 4.1573 | **-0.2289** |

Inside the 8192 window TTT changes nothing (+0.002 at most): attention already has the
context, so there is nothing to recover. Loss jumps from 2.27 to 4.21 the moment the
window starts discarding tokens, and that is exactly where TTT pays - increasingly so
with distance.

**This is the opposite of the paper's Figure 6**, which finds the advantage concentrated in
EARLY tokens. The difference is explainable: their W0 is meta-learned from scratch, so their
initialisation itself is better everywhere, whereas ours is a frozen pretrained Llama whose
only benefit from TTT is the memory mechanism. Our curve isolates that mechanism cleanly.

## Context scaling: the benefit grows with context length

Same model and window (k=8192) throughout. **The 8K row is NOT matched to the other two**:
it is DCLM rather than PG-19, and its arm B value is the inner-lr **2e-5** entry of the 8K
scan (no 8K run at 4e-6 exists). The 16K and 32K rows use PG-19 at 4e-6. An earlier
version of this caption claimed one inner rule and one dataset for all three rows; that was
wrong (caught in the 2026-09-20 check-in review).

| context T | data | inner lr | T/k | arm A (no TTT) | arm B (TTT) | delta |
|---|---|---|---|---|---|---|
| 8192 | DCLM | 2e-5 | 1 | 2.4940 | 2.6632 | +0.1692 |
| 16384 | PG-19 | 4e-6 | 2 | 2.9764 | 2.9286 | **-0.0478** |
| 32768 | PG-19 | 4e-6 | 4 | 3.7119 | 3.5694 | **-0.1424** |

So "+0.169 -> -0.048 -> -0.142" is not a clean trend in T alone: at the same 2e-5, TTT
also hurts at 32K (+0.0670, table above), which means part of the 8K harm is an over-large
step rather than the context regime. The matched evidence for the T/k = 1 regime is the
in-window band of the 32K evaluation (same data, same 4e-6): TTT changes the loss on
positions 0 to 8K by -0.0013, i.e. it does nothing when the context is inside the window,
and the gain appears only beyond it (+0.1132, +0.2041, +0.2550 per 8K band; see the
2026-09-20 section). A matched 8K PG-19 pair at 4e-6 has not been run.

Test-time training goes from harmful to helpful to more helpful as more of the context
falls outside the window. This is the qualitative behaviour TTT-E2E's Figure 1 reports -
their method keeps its advantage as context grows while RNN baselines lose theirs - and we
reproduce it here from a frozen pretrained Llama with no meta-learning at all.

The 8K row is a different dataset (DCLM rather than PG-19) because 8K is the pre-training
stage in the paper's protocol, so its absolute loss is not comparable to the other two rows.
Its sign is what matters: at T/k = 1 the window is not a bottleneck, sliding-window
attention is full attention, and TTT can only add noise.

## Arm C at 32K: the first run, and what test-time training contributes

Arm C could not run at 32K until 2026-09-19 (every attempt ran out of memory; the fixes
are in FINDINGS section 13). Configuration: PG-19, T = 32768, k = 8192, b = 1024,
`fast_blocks=4`, normalized SGD at `lr_rms = 4e-6`, LoRA r = 64, outer lr 4e-4,
131,072 tokens per step (4 sequences), `truncate_bptt=2`, evaluated on the same 32
shuffled validation sequences as arms A and B. Loss is nats per token.

| run | training | inner loop at eval | loss |
|---|---|---|---|
| arm A | none | off | 3.7119 |
| arm B | none | on | 3.5694 |
| plain LoRA fine-tune (`--inner-lr 0`) | 10 steps | off | 2.7125 |
| **arm C** | 10 steps, through the inner loop | on | **2.6690** |

### What the `--inner-lr 0` run is, and is not

It is tempting to read the table as "0.999 nats from LoRA, 0.044 from TTT". That reading is
wrong, and the reason is the design itself. The LoRA is not a fine-tuning baseline with
test-time training added on top: it IS the slow weight set, and it is meta-learned through
the inner loop so that the large fast-weight updates become useful. With the inner loop
disabled, `W_i = W_0` for every chunk, the meta-gradient collapses to ordinary next-token
loss, and the run learns a different, plainly fine-tuned LoRA. The two runs do not share
slow weights, so nothing can be subtracted.

What the pair does establish is a system-level comparison: at an equal budget of 10 steps,
the full method beats plain LoRA fine-tuning by 0.044 nats (one pair of runs, no seeds,
no interval). It also shows that most of the distance from arm A is something plain
fine-tuning achieves too, which is the honest context for the headline number. (This
paragraph originally called that "adaptation to PG-19". That was wrong: the fine-tune does
not improve the in-window loss at all. See "The un-tuned model falls off a cliff at the
window edge" in the 2026-09-20 section.)

### Same weights, inner loop on and off

The comparison that isolates test-time training holds the trained slow weights fixed and
switches only the inner loop (`--eval-ttt-off`: a second evaluation in the same process
with `lr_rms = 0`). Run `C_32k_abl`, 8 training steps:

| inner loop at eval | loss |
|---|---|
| on | 2.7099 |
| off, identical weights | 2.7370 |
| **difference** | **+0.0272 nats** (2.8% perplexity) |

The 32 evaluation sequences come from 22 distinct PG-19 books (at most 3 from one book),
so they are not independent. Averaging the paired difference within each book and testing
across books (`scripts/paired_ttt_effect.py`): **mean +0.0248, se 0.0026, t = 9.67,
95% CI [+0.0194, +0.0301], positive in 22 of 22 books.** The naive per-sequence version
(mean +0.0272, t = 10.63, CI [+0.0220, +0.0324], 32 of 32) is slightly larger and narrower,
as expected; the per-book figure is the one to quote.

By position, with the 8192-token window marked:

| token positions | inner loop on | off | difference |
|---|---|---|---|
| 0 - 8K (inside the window) | 2.3788 | 2.3849 | +0.0061 |
| 8K - 16K | 2.8353 | 2.8538 | +0.0185 |
| 16K - 24K | 2.8122 | 2.8519 | +0.0396 |
| 24K - 32K | 2.8135 | 2.8580 | +0.0444 |

The gain grows monotonically with distance past the attention window, which is what fast
weights acting as a compressed memory of out-of-window context should look like. Arm B,
which has no LoRA at all, shows the same shape.

The small positive difference inside the window (+0.0061) is worth a note. Between the two
separately trained runs above that band differed by -0.0006, i.e. nothing. Slow weights
trained through the inner loop are slightly worse when it is switched off, which would be
expected if they had specialised toward steering the fast-weight updates. At 0.006 nats
this is suggestive only.

### Caveats

- **Short training.** 8 to 10 outer steps. Step ladders (20, 60, 150) and a batch ladder
  (8, 16, 32 sequences per step) are queued; each carries the same on/off evaluation.
- **Truncated meta-gradient.** `truncate_bptt=2` means the meta-gradient spans 2 inner
  steps out of 32. It is what makes 32K fit on one 80 GiB card, and it is a biased
  estimator (PERK, arXiv:2507.06415, accepts the same bias). A `truncate_bptt=1` run is
  queued to measure how much the window length matters.
- **Budget versus the paper.** TTT-E2E's 760M 32K extension uses 32 sequences per step
  (1,048,576 tokens) for far more steps than this, and Books3 where we use PG-19 (the free
  alternative). These numbers are not comparable to their published tables.
- **`peak_gib` in the result files is the training peak**, not the evaluation peak: it is
  read once at the end of the process. Evaluation at 32K peaks at 11.5 GiB.

## 2026-09-20: TTT alone with a per-book interval, and what the 32K baseline really is

### TTT alone (arm B against arm A), nothing trained

Arms A and B were re-scored at 32K with per-sequence losses recorded (login-node H100;
A = 3.7119, identical to the earlier A100 value; B = 3.5691 against 3.5694 earlier).
`scripts/paired_ttt_effect.py B_32k_perseq.json --baseline A_32k_perseq.json`, clustered by
book (32 sequences, 22 books):

**+0.1405 nats, se 0.0061, t = 23.2, 95% CI [+0.1279, +0.1531], positive in 22 of 22 books.**
(Per sequence: +0.1428, t = 25.3, 32 of 32.)

| token positions | arm A (no TTT) | arm B (TTT) | difference |
|---|---|---|---|
| 0 - 8K (inside the window) | 2.3279 | 2.3292 | -0.0013 |
| 8K - 16K | 4.2052 | 4.0920 | +0.1132 |
| 16K - 24K | 4.1841 | 3.9800 | +0.2041 |
| 24K - 32K | 4.1308 | 3.8757 | +0.2550 |

### The un-tuned model falls off a cliff at the window edge

Arm A scores 2.33 inside the 8192-token window and 4.13 to 4.21 beyond it. That is a cliff,
not a gradual loss of context. The plain LoRA fine-tune (`--inner-lr 0`, 10 steps) scores
2.3754 in the window and about 2.83 beyond it: fine-tuning made the in-window loss slightly
WORSE (+0.05) and the beyond-window loss about 1.35 nats better.

So the earlier description of arm C's gain over arm A as "adaptation to PG-19" was wrong.
Adaptation to the domain would help inside the window too, and it does not. The working
hypothesis was that a Llama pretrained with full attention breaks under a sliding window
once its earliest tokens leave the window, and that fine-tuning repairs this. The
diagnostic, arm A at 32K with full attention (`--window 32768`), has since been run and
CONFIRMS that the collapse is caused by the sliding window: see "The full-attention
diagnostic" below. (Which property of the window breaks the model, for instance losing the
first tokens as attention sinks, has not been tested.)

### What TTT-E2E's protocol says the baseline is

From the reference repository (`configs/training/760m/ext.yaml`,
`configs/experiment/760m/extension/`): every model in their 32K comparison is first
extension-trained at 32K for 725 steps at 32 sequences per step on Books3 (about 760M
tokens), including the full-attention baseline (`ext-760m-fa-32K.yaml`, which also raises
`rope_theta` to 2,000,000). Their outer settings equal ours (lr 4e-4, 10% warmup, end lr
1e-5, weight decay 0.1, beta2 0.95). Their pretraining uses SWA with an 8192 window.

Consequences for reading the tables above:
- Un-tuned arms A and B are outside that protocol. They remain useful as a mechanism probe
  (TTT helps only beyond the window, with nothing trained), not as the baseline.
- The protocol-faithful baseline is an extension-trained model without TTT, which is what
  the `--inner-lr 0` control is. Arm C should be compared against it at equal budget:
  2.7125 against 2.6690 at 10 steps; the 20-step pair is queued.
- Our extension budget (10 to 20 steps x 131,072 tokens = 1.3M to 2.6M tokens) is roughly
  300 to 600 times smaller than theirs. Matching it would cost about 212 A100-hours on one
  GPU at the measured 33 s per sequence. Not decided.
- `--inner none` was checked to be bit-identical to `--inner-lr 0` on SmolLM2-135M (all 10
  steps and the evaluation, difference exactly 0), so longer controls can skip the double
  backward. On real hardware at 32K (`C_32k_ctl_none10`, job 14195736, against
  `C_32k_ctl_lr0`): largest per-step loss difference 2.5e-4, i.e. the noise floor;
  evaluation 2.71244 against 2.71249; and 68.7 against 131.3 seconds per step, 1.9x faster.

### AdamW as the inner optimizer at 32K (arm B, nothing trained)

beta1 = beta2 = 0.9, eps = 1e-8, warm start from the first chunk's gradient.

| inner lr | loss | vs arm A (3.7119) |
|---|---|---|
| 1e-6 | 3.7021 | -0.0098 |
| 2e-6 | 3.6858 | -0.0261 |
| 4e-6 | 3.6597 | -0.0522 |
| 7e-6 | 3.6369 | -0.0750 |
| 2e-5 | 3.6175 | -0.0944 |

Normalized SGD at the same per-element step (4e-6) reaches 3.5691, so at equal step size
AdamW recovers about a third as much. AdamW was still improving at 2e-5, the top of this
grid, where normalized SGD already hurts (3.7788); its optimum is bracketed in the
extension below.

### Checkpoint / resume on real hardware

`scripts/della/resume_check.sbatch` (job 14174410, A100): the 10-step arm C configuration
was killed with SIGKILL after 4 checkpoint writes (exit 137) and the identical command
resumed at step 4/10. Against the uninterrupted run `C_32k_q10` (job 14163396):

- evaluation 2.66892 against 2.66895;
- loss differences per step: 0, 0, 9.8e-5, 1.9e-5 before the kill and 1.0e-4, 2.3e-5,
  1.7e-4, 9.1e-5, 3.0e-5, 7.4e-5 after the resume, i.e. no change at the boundary;
- the first difference of any kind is the gradient norm at step 0 (6.69440 against
  6.69432), before any checkpoint existed. The two runs were on different nodes
  (della-l08g2, della-l07g2), so this is kernel nondeterminism. Losses stay identical
  through step 1 because the outer learning rate at step 0 is exactly 0.
- The largest gradient-norm difference is 1.7% at step 8. That is within run-to-run noise:
  `C_32k_q10_repeat` (job 14194873), a second UNINTERRUPTED run of the same configuration,
  differs from the reference by up to 2.49e-4 in loss (mean 7.8e-5), by 1.2% in gradient
  norm, also at step 8, and by 1.4e-4 in evaluation loss (2.66909 against 2.66895). The
  killed-and-resumed run differs by LESS on loss (max 1.73e-4, mean 6.1e-5) and on
  evaluation (3e-5). So two runs of this configuration agree to about 2.5e-4 nats, resumed
  or not; differences of that size between runs carry no information.

On CPU the same procedure through the real CLI is bit-identical (SmolLM2-135M, 10 steps x 8
metrics and both evaluations per token).

## 2026-09-20, later: the sliding window breaks the un-tuned model

Tables in this section are printed by `scripts/render_tables.py` from the result files; each
row's settings are read from that file's own arguments, not typed.

### The full-attention diagnostic

| run | settings (from the file) | 0-8K | 8-16K | 16-24K | 24-32K | overall |
|---|---|---|---|---|---|---|
| A_32k_perseq | T=32768 k=8192 none lr=- n=32 | 2.3279 | 4.2052 | 4.1841 | 4.1308 | 3.7119 |
| A_32k_fullattn | T=32768 k=32768 none lr=- n=32 | 2.3279 | 2.3119 | 2.2792 | 2.3183 | 2.3092 |
| B_32k_perseq | T=32768 k=8192 normalized_sgd lr=4e-06 n=32 | 2.3292 | 4.0920 | 3.9800 | 3.8757 | 3.5691 |
| B_32k_fullattn | T=32768 k=32768 normalized_sgd lr=4e-06 n=32 | 2.3293 | 2.3138 | 2.2802 | 2.3192 | 2.3105 |

The same un-tuned weights, scored with full attention instead of the 8192-token window, show no
cliff: about 2.30 everywhere. The first band is identical in both, as it must be (inside the
window the two are the same computation). Beyond the window the windowed model is about 1.85
nats worse than itself with full attention. So the collapse is caused by the sliding window,
not by missing information: a Llama pretrained with full attention does not function when its
attention is cut to the last 8192 tokens. TTT on top of full attention does nothing (2.3105
against 2.3092), the T/k = 1 behaviour. Which property of the window breaks the model has not
been tested.

This changes how every arm B number must be read. Arm B's gain sits entirely beyond the window,
which was taken as the signature of fast weights acting as a compressed memory. It is equally
the signature of fast weights compensating for a broken model, because the breakage also exists
only beyond the window. The position pattern does not separate the two readings; the size of
what out-of-window context is worth to a HEALTHY model does (next subsection but one).

### The matched T/k = 1 row

PG-19 at T = 8192, the same inner rule as the 32K row (normalized SGD, 4e-6), 128 sequences
from 50 books. This replaces the unmatched 8K row of "Context scaling" above.

| T | T/k | arm A | arm B | B - A | source files |
|---|---|---|---|---|---|
| 8192 | 1 | 2.3067 | 2.3078 | +0.0012 | A_8k_pg19, B_8k_pg19 |
| 32768 | 4 | 3.7119 | 3.5691 | -0.1428 | A_32k_perseq, B_32k_perseq |

`scripts/paired_ttt_effect.py B_8k_pg19.json --baseline A_8k_pg19.json`, clustered by book: TTT
changes the loss by -0.0016 nats per book (positive = TTT helps), se 0.0002, t = -7.33, 95% CI
[-0.0020, -0.0011], positive in 6 of 50 books. When the whole context is inside the window the
inner loop is slightly but significantly harmful. This agrees with the in-window band of the
32K evaluation (-0.0013).

### AdamW: the full curve

| inner lr | loss | vs arm A (3.7119) |
|---|---|---|
| 1e-06 | 3.7021 | -0.0098 |
| 2e-06 | 3.6858 | -0.0261 |
| 4e-06 | 3.6597 | -0.0522 |
| 7e-06 | 3.6369 | -0.0750 |
| 2e-05 | 3.6175 | -0.0944 |
| 5e-05 | 3.6623 | -0.0495 |
| 0.0001 | 3.7715 | +0.0596 |
| 0.0002 | 3.9852 | +0.2733 |

AdamW's optimum as arm B is bracketed at about 2e-5 (-0.0944), five times the step normalized
SGD prefers (4e-6, -0.1428), and its best is about two thirds of normalized SGD's best. Above
5e-5 it is worse than no TTT. No meta-trained run with AdamW as the inner rule exists.

### What context beyond the window is worth

`scripts/context_value.py` scores the same tokens twice with the un-tuned model, both times
with a healthy full-attention forward: once over the whole 32K sequence (FULL), once with the
context restarted every S = 8192 tokens (RESTART: independent 8192-token segments, positions
restarting at 0). For a token in segments 1 to 3, NLL_restart - NLL_full is what all the older
context is worth to a model that still has q recent tokens. Paired per sequence, clustered by
book (32 sequences, 22 books).

| window S | recent context in the restart condition | full | restart | value (per document) | 95% CI | documents positive |
|---|---|---|---|---|---|---|
| 8192 | >= 4096 tokens | 2.3187 | 2.3487 | +0.0304 | [+0.0220, +0.0388] | 21/22 |
| 8192 | >= 6144 tokens | 2.3281 | 2.3505 | +0.0224 | [+0.0154, +0.0295] | 21/22 |
| 8192 | >= 7168 tokens | 2.3187 | 2.3395 | +0.0208 | [+0.0139, +0.0277] | 21/22 |
| 8192 | sanity: segment 0, identical input | | | max abs diff 0.0e+00 | | |
| 2048 | >= 1024 tokens | 2.2990 | 2.3756 | +0.0790 | [+0.0658, +0.0922] | 22/22 |
| 2048 | >= 1536 tokens | 2.2978 | 2.3644 | +0.0693 | [+0.0564, +0.0821] | 22/22 |
| 2048 | >= 1792 tokens | 2.2956 | 2.3602 | +0.0667 | [+0.0544, +0.0790] | 22/22 |
| 2048 | sanity: segment 0, identical input | | | max abs diff 0.0e+00 | | |
| 1024 | >= 512 tokens | 2.3014 | 2.4129 | +0.1149 | [+0.0976, +0.1322] | 22/22 |
| 1024 | >= 768 tokens | 2.3038 | 2.4049 | +0.1041 | [+0.0878, +0.1203] | 22/22 |
| 1024 | >= 896 tokens | 2.2920 | 2.3892 | +0.0992 | [+0.0839, +0.1145] | 22/22 |
| 1024 | sanity: segment 0, identical input | | | max abs diff 0.0e+00 | | |

The ceiling rises quickly as the window shrinks. With the tightest comparison in each case (the restart condition keeps at least 7/8 of a window of recent tokens), everything outside the window is worth +0.0208 nats per book at S = 8192 ([+0.0139, +0.0277]), +0.0667 at S = 2048 ([+0.0544, +0.0790]) and +0.0992 at S = 1024 ([+0.0839, +0.1145]). A smaller window therefore gives an out-of-window memory several times more to gain, on the same data and model. The reference uses k = 8192; the constraint k >= b still holds at S = 1024 with the 1024-token chunk.

The sanity row is exact: given identical input the two code paths agree to the last bit, so the
differences above are due to context length alone.

**On this benchmark, everything outside an 8192-token window is worth about 0.02 nats per
token to a healthy model.** That is an UPPER bound for a sliding window, which always keeps
8192 recent tokens, more than the restart condition's 7168 or more. It is the most any
out-of-window memory mechanism, test-time training included, can gain at T = 32768, k = 8192 on
PG-19 with this model. Against it:

- arm B's gain over arm A is +0.1405 nats per book. At most about 0.02 of that can be recovered
  long-range information; the rest, at least 85%, is repair of the damage the window does to
  the un-tuned model.
- the same-weights gain from TTT on the 8-step arm C model, +0.0248, is about the size of the
  ceiling, but it was measured on a model that is still broken beyond the window (about 2.83
  there against about 2.32 healthy), so it cannot be credited to memory either.

Three of the 32 evaluation sequences are near-memorised text (0.13, 0.17 and 0.24 nats under
full attention; one book). They are scored identically in every arm and count as one book in
the clustered tests.

### Arm D: the slow set was empty, and what it costs to run

`slow_spec=("**",)` was matched as a substring, and "**" occurs in no parameter name, so arm D
had an EMPTY slow set and had never been runnable. The first arm D memory probe printed
`slow params: 0` and is void. "**" is now a wildcard for every non-fast parameter (the fast
rule wins; nothing is frozen; it is refused if mixed with patterns), tested on the split and
with a real training step under per-window truncated BPTT. On Llama-3.2-1B the split is fast
201,326,592 / slow 1,034,487,820 / frozen 0 (the `[run]` line of jobs 14198617 and 14198618).
Two properties of this arm as implemented: the fast weights' initialisation W0 is NOT trained
(the split is disjoint and the outer optimizer owns only the slow set), and no weight decay is
applied (decay is defined on the LoRA factors only, and arm D has none).

**Arm D does not fit one 80 GiB GPU at 32K with `truncate_bptt=2`.** Validation runs at outer
learning rates 4e-4 and 4e-5 (jobs 14198617, 14198618; 10 steps, 4 sequences per step) completed
step 0 (loss 3.7461, gradient norm 33.96 over all weights, 129 seconds per step, the same speed
as arm C) and ran out of memory in step 1 with 78.73 GiB in use: arm C's 68 GiB training peak
plus the gradients and the two AdamW moments of 1.03 B parameters. Step 0 fits only because
AdamW allocates its moments at the first optimizer step.

Validation at `truncate_bptt=1` (jobs 14198690 and 14198691; 10 steps, 4 sequences per step,
115 seconds per step): **arm D fits one 80 GiB GPU, training peak 71.8 GiB.** Arm C has a
matched run queued at this truncation (`C32k_t1`).

| outer lr | loss at step 5 | loss at step 9 | evaluation (32 sequences) |
|---|---|---|---|
| 4e-4 (arm C's rate) | 7.6948 | 7.2855 | 7.0392 |
| 4e-5 | 3.2999 | 2.9330 | 2.7306 |

Full fine-tuning of the pretrained model at the LoRA learning rate DIVERGES (the loss more than
doubles by step 5). At 4e-5 it trains normally and reaches 2.7306, which is worse than arm C at
the same step count (2.6690) and worse than the plain LoRA fine-tune (2.7125). This is a
feasibility run, not a test of H1: it differs from arm C's run in truncation (1 against 2) and
in learning rate, 4e-5 may simply be too cautious for 10 steps, and as implemented arm D does
not train W0. A fair arm D needs its own learning-rate sweep at `truncate_bptt=1`.

### The 2x2 behind H1

H1 is about slow weights meta-learned THROUGH the inner loop. At a fixed 10-step budget, cross
how the slow weights were trained with whether TTT runs at evaluation. `--load-slow` evaluates
saved slow weights under a chosen inner rule; each run below also re-evaluated its diagonal
cell, which reproduced the number already on record (2.6688 for 2.66892; 2.7124 for 2.71244),
so the right weights were loaded. Weights: `C_32k_resumecheck.ckpt` (trained with
`--inner normalized_sgd`, 4e-6) and `C_32k_ctl_none10.ckpt` (trained with `--inner none`).

| slow weights | TTT on at eval | TTT off at eval |
|---|---|---|
| trained through the inner loop | 2.6688 | 2.6888 |
| plain fine-tune | 2.6944 | 2.7124 |

`scripts/two_by_two.py`, per document (32 sequences, 22 books), positive = lower loss:

| effect | per document | 95% CI | t | documents positive |
|---|---|---|---|---|
| TTT at eval, on meta-learned weights | +0.0179 | [+0.0131, +0.0227] | 7.72 | 22/22 |
| TTT at eval, on plain fine-tuned weights | +0.0164 | [+0.0121, +0.0206] | 8.07 | 22/22 |
| training through the inner loop, TTT on | +0.0247 | [+0.0219, +0.0275] | 18.59 | 22/22 |
| training through the inner loop, TTT off | +0.0232 | [+0.0204, +0.0260] | 17.18 | 22/22 |
| INTERACTION: does training through the inner loop make TTT more useful? | +0.0015 | [+0.0005, +0.0026] | 2.95 | 17/22 |

The two effects are close to additive. Test-time training is worth about the same on either set
of weights, and training through the inner loop gives better slow weights almost equally with
TTT switched OFF. The interaction, which is the quantity H1 is about, is distinguishable from
zero but is +0.0015 nats: about a thirtieth of arm C's lead over the plain fine-tune. Why
training through the inner loop helps with TTT off is NOT explained. Caveats: one training run
per row (no seeds; re-running one configuration moves the evaluation by about 2.5e-4, far less
than these effects, but seed-to-seed variation is unmeasured); 10 steps; and both models are
still about half a nat from healthy beyond the window, so all four cells sit in the repair
regime of the previous subsections.

## Runs in flight (as last observed 2026-09-20, about 14:25 ET)

On 2026-09-20 `sbatch --test-only` estimated a start of 2026-09-24 for any job longer than
61 minutes; jobs of at most 61 minutes (`--qos=gpu-test`) start within minutes. All long
jobs below are pending, constrained to 80 GiB GPUs, and resumable (checkpoint after every
step, path derived from `--out`). All are arm C at 32K (PG-19, k = 8192, b = 1024,
`fast_blocks=4`, normalized SGD 4e-6, outer lr 4e-4, LoRA r = 64, `truncate_bptt=2` unless
noted) and evaluate the same 32 validation sequences with the inner loop on and off.

| job | name | what it varies | purpose |
|---|---|---|---|
| 14163397 | C32k_t2 | 20 steps, 4 seq/step | the properly warmed-up arm C number |
| 14163398 | C32k_t1 | same, `truncate_bptt=1` | how much the truncation window matters |
| 14165376 | C32k_ctl20 | same as t2 with `--inner-lr 0` | budget-matched extension-only baseline |
| 14169727 | C32k_s60 | 60 steps | step ladder: is arm C undertrained? |
| 14169728 | C32k_s150 | 150 steps | step ladder |
| 14167119 | C32k_bs8 | 8 seq/step, 20 steps | batch ladder: is the meta-gradient noise-limited? |
| 14167120 | C32k_bs16 | 16 seq/step, 20 steps | batch ladder |
| 14167121 | C32k_bs32 | 32 seq/step, 20 steps | batch ladder; matches the reference batch (1,048,576 tokens) |
| 14169729 | C32k_bs32s60 | 32 seq/step, 60 steps | closest approach to the reference regime (63M tokens) |

Wall-time limits as submitted: C32k_t2, C32k_t1, C32k_ctl20, C32k_bs16 5 h; C32k_s60,
C32k_bs8 4 h; C32k_s150, C32k_bs32 8 h; C32k_bs32s60 22 h. At the measured 132 s per step
for 4 sequences (33 s per sequence) the longest training phases are about 5.5 h (s150),
5.9 h (bs32) and 17.6 h (bs32s60), inside their limits; a job that does hit its limit
resumes from its last step.

All 32K arm C runs so far used the CLI default `--lora-targets wq,wk,wv,wo,w1,w2,w3`, i.e.
LoRA on the attention projections AND on the MLPs, 45,156,364 slow parameters (the `[run]`
line of each job log). The 8K pilot and the local SmolLM2 run used attention-only LoRA.

The batch ladder holds steps fixed, so larger batches also see more tokens; it measures
"more compute per step", not batch size in isolation. The 60- and 150-step runs do not yet
have budget-matched `--inner none` controls queued.

The short runs that were unread when this list was first written (`C_32k_q10_repeat`,
`C_32k_ctl_none10`, the AdamW grid, `A_32k_fullattn`, `B_32k_fullattn`, the arm D probe) have
all been read; their results are in "2026-09-20, later" above and in "Checkpoint / resume on
real hardware".

Not started: arm D beyond its 10-step feasibility run (it needs a learning-rate sweep at
`truncate_bptt=1`), arm F (not implemented on the Llama path; `--arm F` now refuses to run), the
forgetting probe on the real model (wired into the CLI as `--forgetting-probe-tokens`,
exercised only on SmolLM2-135M), the decay-toward-W0 sweep, the LoRA rank sweep, Muon, and
multiple seeds.
