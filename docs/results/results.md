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

## Round 2 plan (2026-09-21): where the thesis has room to show itself

Decided by Helen after the check-in: try a smaller window (k = 1024) with arm C; use a longer
context or a task with real long-range dependence; try another dataset (SlimPajama); and match
the reference's 32K extension budget. Nothing in this section is a result yet.

**Matched extension budget (k = 8192, the reference protocol).** 725 steps at 32 sequences per
step, about 760M tokens: arm C (about 212 GPU-hours at the measured 33 s per sequence) and the
extension-only control it has to be compared with (`--inner none`, about 111 GPU-hours). Both
exceed every wall limit on one GPU (24 h, 72 h, 144 h), and a chain of dependent links pays a
queue wait PER LINK, because Slurm ages a dependent job only once its predecessor ends. So the
sequences of a step are sharded over 4 GPUs (`ttt/train/distributed.py`,
`scripts/della/run_arm_ddp.sbatch`): about 53 h and 28 h, one job each, with a spare link.
Before submitting: a 2-GPU validation of the 4-sequence configuration against `C_32k_q10`
(same global batch, so it must agree within the 2.5e-4 noise floor).

**k = 1024** (`scripts/della/round2_login.sh`, then short validation jobs). Out-of-window
context is worth +0.0992 nats per book at this window against +0.0208 at 8192. First, with
nothing trained: arm A, and an arm B inner-LR scan, since the optimum may move; and arm C's
training memory at `truncate_bptt` 2, 4 and 8, because a smaller window shrinks the per-window
backward spike and may afford a longer, less biased truncation window. Then arm C and its
`--inner none` control at 10 steps, the 2x2 of "The 2x2 behind H1", and longer runs sized from
those timings.

**Another dataset: SlimPajama** (`DKYoon/SlimPajama-6B`, free and ungated; verified 2026-09-21
that rows carry `meta.redpajama_set_name`). `scripts/della/round2_data.sh` keeps documents of at
least 32,769 tokens together with their source domain, using an exact byte-length prefilter so
that the tokenizer is skipped for nearly every web document. The first question is where
long-range context is worth the most: `scripts/context_value.py --only-label` per domain
(books, arXiv, GitHub, ...). Training goes where that ceiling is highest.

**Longer context: PG-19 at 128K.** Only books of at least 131,073 tokens are kept, so a sequence
never spans two books. Same first question: what is context beyond the window worth at 128K?

**A task with real long-range dependence (proposed, not built).** Fact recall beyond the window:
insert a sentence stating a random fact early in a sequence, query it more than k tokens later,
and score the answer tokens with the fact PRESENT against ABSENT. For a sliding window without
test-time training that difference is zero by construction once the distance exceeds L x k
(stacked windowed layers pass information back one window per layer: 16 x 1024 = 16,384 tokens
at k = 1024, more than T at k = 8192, where the no-TTT run measures the floor instead). Full
attention gives the ceiling, and whatever test-time training recovers is memory and nothing
else: repair of the broken window helps all tokens alike and cancels in the difference.

## 2026-09-21: window k = 1024, and data parallelism

All on PG-19 at T = 32768 with the same 32 validation sequences (22 books) as every earlier
32K result; chunk b = 1024, so k >= b still holds, with equality.

### Nothing trained, k = 1024

| run | inner lr | loss |
|---|---|---|
| arm A (no TTT) | - | 4.9895 |
| arm B (normalized SGD) | 2e-6 | 4.6065 |
| arm B | 4e-6 | 4.5232 |
| arm B | 7e-6 | 4.5199 |
| arm B | 2e-5 | 4.7039 |
| arm B | 5e-5 | 6.9575 |

The un-tuned model is far more broken at this window (4.9895 against 3.7119 at k = 8192 and
2.3092 with full attention), TTT alone recovers more (0.47 nats at the best rate), and the
inner-LR optimum has not moved: 4e-6 to 7e-6, diverging by 5e-5. 4e-6 is kept so that windows
can be compared at one inner rule. As at k = 8192, most of what TTT recovers here is repair of
the broken window, not memory: the ceiling for memory at this window is +0.0992 nats.

### A smaller window affords a longer truncation window

`scripts/memory_probe.py --window 1024` (one sequence, no optimizer step):

| `truncate_bptt` | peak at k = 1024 | peak at k = 8192 |
|---|---|---|
| 2 | 24.70 GiB | 49.57 GiB |
| 4 | 41.38 GiB | out of memory |
| 8 | 74.86 GiB | - |

So at k = 1024 the meta-gradient can span 4 inner steps instead of 2 on one 80 GiB GPU. The
real 10-step training run at `truncate_bptt=4` peaked at 42.0 GiB (its result file), close to
the probe; the 18 GiB gap between probe and trainer seen at k = 8192 does not appear here. A
step takes 55 seconds (4 sequences) against 132 at k = 8192; the `--inner none` control, 30.

### Arm C at k = 1024, 10 steps, and the 2x2 behind H1

`truncate_bptt=4`, otherwise the settings of the k = 8192 runs (normalized SGD 4e-6, outer lr
4e-4, LoRA r = 64 on attention and MLPs, 4 sequences per step). The plain fine-tuned weights
were evaluated with TTT through `--load-slow`; their TTT-off evaluation reproduced the
control's own number (3.0682), so the right weights were loaded.

| slow weights | TTT on at eval | TTT off at eval |
|---|---|---|
| trained through the inner loop | 2.9440 | 3.0041 |
| plain fine-tune | 3.0212 | 3.0682 |

`scripts/two_by_two.py`, per document (32 sequences, 22 books), positive = lower loss:

| effect | k = 1024 | 95% CI | documents positive | k = 8192 (from "The 2x2 behind H1") |
|---|---|---|---|---|
| TTT at eval, on meta-learned weights | +0.0559 | [+0.0467, +0.0651] | 22/22 | +0.0179 |
| TTT at eval, on plain fine-tuned weights | +0.0433 | [+0.0354, +0.0513] | 22/22 | +0.0164 |
| training through the inner loop, TTT on | +0.0772 | [+0.0703, +0.0841] | 22/22 | +0.0247 |
| training through the inner loop, TTT off | +0.0646 | [+0.0560, +0.0733] | 22/22 | +0.0232 |
| INTERACTION: does training through the inner loop make TTT more useful? | +0.0126 | [+0.0098, +0.0153] | 22/22 | +0.0015 |

Every effect is about three times its k = 8192 value, and the interaction, the quantity H1 is
about, is about eight times larger and positive in all 22 books (17 of 22 at k = 8192): at this
window test-time training is worth 29% more on weights that were trained through the inner
loop than on plainly fine-tuned ones. This is the first evidence in the project FOR H1's
mechanism. It is early evidence: 10 steps, one training run per row and no seeds, a truncation
window (4) that differs from the k = 8192 runs (2), and models that are still about 0.6 nats
from the healthy level for this window (about 2.39, from "What context beyond the window is
worth"), so all four cells still sit in the repair regime. A 40-step pair was submitted to see
whether the interaction grows with training.

### Data parallelism over sequences

The sequences of an outer step are independent, so the step gradient splits across GPUs
exactly (`ttt/train/distributed.py`): in float64 two processes equal one to 1e-12, and two
ranks killed with SIGKILL and resumed are bit-identical to two ranks left alone. On real
hardware, 2 GPUs x 2 sequences against the single-GPU run `C_32k_q10` (same global batch):

- largest per-step loss difference 3.737e-04, the same order as the 2.49e-4 between two
  single-GPU runs; evaluation 2.668930 against 2.668952;
- 65.7 seconds per step against 131.9: a factor of 2.0 on 2 GPUs.

The first attempt (job 14236036) trained correctly and was then KILLED during evaluation: srun
terminates the remaining tasks 60 seconds after the first task exits, and ranks other than 0
exit once training is done. `srun --wait=0` fixes it. No local test can see launcher behaviour;
the validation job did, and the orchestrator's gate withheld the 320 GPU-hour submission.

### Where data preparation runs

Tokenisation on the login node was killed three times (exit 137) after about 10 minutes of
CPU, niced and capped at 4 threads notwithstanding. The login node now only downloads the
parquet shards (21 GB in 3.5 minutes); `ttt.data.prepare --data-files` reads them offline in CPU
compute jobs, which started within a second of submission.

## 2026-09-21, afternoon: 40 steps at k = 1024, other domains, and 128K

### Arm C at k = 1024, 40 steps: does the interaction grow with training?

Same settings as the 10-step pair (PG-19, T = 32768, `truncate_bptt=4`, 4 sequences per step,
outer lr 4e-4, normalized SGD 4e-6), 40 steps. Prediction written down before the last cell was
evaluated: the interaction stays near +0.01 (between +0.005 and +0.02) while the TTT effect
itself shrinks, because the LoRA takes over the repair that TTT was doing.

| slow weights | TTT on at eval | TTT off at eval |
|---|---|---|
| trained through the inner loop (job 14237563, 41 minutes) | 2.6777 | 2.7086 |
| plain fine-tune (job 14237564, 23 minutes) | 2.6979 | 2.7196 |

The plain weights were evaluated with TTT through `--load-slow`; their TTT-off evaluation
reproduced the control's own 2.7196. Per document (32 sequences, 22 books), positive = lower loss:

| effect | 10 steps | 40 steps | 95% CI at 40 steps | documents positive | change |
|---|---|---|---|---|---|
| TTT at eval, on meta-learned weights | +0.0559 | +0.0278 | [+0.0210, +0.0347] | 22/22 | -50% |
| TTT at eval, on plain fine-tuned weights | +0.0433 | +0.0195 | [+0.0143, +0.0248] | 22/22 | -55% |
| training through the inner loop, TTT on | +0.0772 | +0.0179 | [+0.0137, +0.0220] | 22/22 | -77% |
| training through the inner loop, TTT off | +0.0646 | +0.0096 | [+0.0060, +0.0132] | 21/22 | -85% |
| INTERACTION (what H1 is about) | +0.0126 | +0.0083 | [+0.0062, +0.0104] | 22/22 | -34% |

In nats the interaction did not grow: it fell by a third, inside the predicted range. Everything
else fell faster, as training moved the model towards the healthy level for this window (the
control went from 3.0682 to 2.7196; healthy is about 2.39). What TTT is worth halved on both
sets of weights. What training through the inner loop is worth with TTT OFF fell by 85%: that
part was a head start on repairing the window, and the plain fine-tune is catching up. The
interaction is the most persistent of the five effects and is positive in all 22 books at both
lengths. Relative to what TTT is worth on plainly fine-tuned weights it rose from 29% to 43%;
as a share of what training through the inner loop buys with TTT on, from 16% to 46%.

It remains small, +0.0083 nats or 0.3% of the loss, on models still about 0.3 nats above the
healthy level, so both rows are still in the repair regime. Language-model loss cannot say
whether the interaction is memory or a repair that has learned to rely on TTT. The matched
budget (queued) removes the repair regime; a probe in which repair cancels by construction
(see "A task with real long-range dependence") is the direct test.

### Where out-of-window context is worth most: SlimPajama by source domain

`DKYoon/SlimPajama-6B` (free, ungated), documents of at least 32,769 Llama-3 tokens, every
10th document held out: 6,176 training documents (476,513,860 tokens) and 687 validation
documents (53,238,246 tokens). Training documents by source: CommonCrawl 2,776, arXiv 1,500,
Book 1,375, GitHub 448, Wikipedia 50, C4 26, StackExchange 1.

`scripts/context_value.py --only-label`, 24 validation sequences per domain, un-tuned model,
healthy full attention in both conditions; tightest band (the restart condition keeps at least
7/8 of S recent tokens). `full` and `restart` are token means; the value is the mean over
documents, so it need not equal their difference exactly.

| corpus, domain | S | full | restart | value (per document) | 95% CI | documents positive |
|---|---|---|---|---|---|---|
| PG-19 (from above) | 8192 | 2.3187 | 2.3395 | +0.0208 | [+0.0139, +0.0277] | 21/22 |
| SlimPajama arXiv | 8192 | 0.9637 | 1.0394 | +0.0814 | [+0.0209, +0.1419] | 20/21 |
| SlimPajama CommonCrawl | 8192 | 2.1249 | 2.1648 | +0.0392 | [+0.0064, +0.0720] | 16/21 |
| SlimPajama Book | 8192 | 2.3557 | 2.3876 | +0.0325 | [+0.0177, +0.0472] | 22/23 |
| SlimPajama GitHub | 8192 | 0.7063 | 0.7361 | +0.0314 | [+0.0134, +0.0494] | 19/21 |
| SlimPajama Wikipedia (8 sequences, 4 documents) | 8192 | 2.1974 | 2.2101 | +0.0099 | [-0.0008, +0.0206] | 4/4 |
| PG-19 (from above) | 2048 | 2.2956 | 2.3602 | +0.0667 | [+0.0544, +0.0790] | 22/22 |
| SlimPajama arXiv | 2048 | 0.9923 | 1.1611 | +0.1752 | [+0.0991, +0.2513] | 21/21 |
| PG-19 (from above) | 1024 | 2.2920 | 2.3892 | +0.0992 | [+0.0839, +0.1145] | 22/22 |
| SlimPajama arXiv | 1024 | 1.0178 | 1.2684 | +0.2548 | [+0.1804, +0.3291] | 21/21 |
| SlimPajama GitHub | 1024 | 0.7572 | 0.9656 | +0.2133 | [+0.1439, +0.2826] | 21/21 |

The sanity check (segment 0, identical input) is exactly 0 in every run. arXiv is where a
memory has the most to gain: 3.9 times PG-19's ceiling at the reference window k = 8192 and
2.6 times at k = 1024, where it is a quarter of the healthy loss (0.2548 of 1.0178) against 4%
on PG-19. Code is close behind at k = 1024. The arXiv intervals are wide: papers differ a lot
in how much they refer back.

Next on this corpus (jobs 14239404 and 14239405, running): arm C and its `--inner none` control
at k = 1024, 40 steps, on the SlimPajama mix, scored on the first 96 validation sequences, with
the context value and the un-tuned window damage measured on the same 96 sequences
(`scripts/della/login_slimpajama_k1024.sh`). The question is whether what TTT gains on a
sequence tracks what out-of-window context is worth there (memory) or how much the window
damages the model there (repair).

### SlimPajama at k = 1024: nothing trained, and the first 40-step pair

First 96 validation sequences of the evaluation order (88 documents: 41 CommonCrawl sequences,
29 Book, 19 arXiv, 7 GitHub), T = 32768, k = 1024 (`scripts/della/login_slimpajama_k1024.sh`).

| run, nothing trained | inner lr | loss |
|---|---|---|
| arm A (no TTT) | - | 4.3094 |
| arm B (normalized SGD) | 2e-6 | 3.9722 |
| arm B | 4e-6 | 3.8821 |
| arm B | 7e-6 | 3.8620 |
| arm B | 2e-5 | 4.0237 |

The inner-LR optimum sits where it did on PG-19 (4e-6 to 7e-6), so 4e-6 is kept. On the same 96
sequences `context_value.py` (S = 1024, four pieces of 24; the new `--skip-sequences` slicing
first reproduced sequences [12, 24) of the arXiv run to under 1e-4) gives the ceiling and the
healthy full-attention loss per sequence, and `scripts/ttt_vs_context_value.py` sets them
against what TTT alone gains. damage = arm A loss - (full + ceiling): how far the windowed
model sits above a healthy model limited to the same window.

| domain | sequences | documents | full attention | ceiling | damage (arm A) | TTT alone (A - B) | gain / damage |
|---|---|---|---|---|---|---|---|
| all | 96 | 88 | 1.9404 | +0.1519 | +2.2171 | +0.4274 | 0.19 |
| arXiv | 19 | 17 | 1.0846 | +0.2500 | +1.5478 | +0.3168 | 0.20 |
| Book | 29 | 28 | 2.4044 | +0.1092 | +2.6285 | +0.5238 | 0.20 |
| CommonCrawl | 41 | 36 | 2.2185 | +0.1186 | +2.3723 | +0.4256 | 0.18 |
| GitHub | 7 | 7 | 0.7121 | +0.2582 | +1.4206 | +0.3386 | 0.24 |

By domain, the un-tuned gain follows the DAMAGE, not the ceiling: TTT recovers about a fifth of
the damage everywhere, and gains most on books, where the ceiling is lowest. Across the 88
documents, gain = b0 + b_ceiling x ceiling + b_damage x damage gives b_damage = +0.215
[+0.184, +0.246] and b_ceiling = +0.438 [+0.273, +0.603] (R2 0.69; ceiling and damage correlate
at -0.35, so they can be told apart). The intercept is -0.116 nats, so the line is a description,
not a structural model. Two checks that could remove the ceiling coefficient did not: adding
the loss level gives +0.421 [+0.243, +0.598]; adding domain indicators gives +0.453
[+0.274, +0.632].

40 steps, same settings as the PG-19 pair (jobs 14239404 and 14239405; 40 and 21 minutes), each
job's own evaluation on the first 32 sequences:

| slow weights | TTT on at eval | TTT off at eval |
|---|---|---|
| trained through the inner loop | 2.3600 | 2.4018 |
| plain fine-tune (`--inner none`) | - | 2.4086 |

All four combinations on the 96 sequences (`scripts/della/login_sp_cells.sh`; the loaded
weights reproduce the jobs' own numbers on the first 32 sequences to 3e-5):

| slow weights | TTT on at eval | TTT off at eval |
|---|---|---|
| trained through the inner loop | 2.2789 | 2.3193 |
| plain fine-tune | 2.2920 | 2.3252 |

| effect, per document (88 documents) | mean | 95% CI | documents positive |
|---|---|---|---|
| TTT at eval, weights trained with TTT | +0.0418 | [+0.0318, +0.0518] | 88/88 |
| TTT at eval, plain fine-tuned weights | +0.0345 | [+0.0257, +0.0432] | 88/88 |
| training with TTT, evaluated with TTT | +0.0134 | [+0.0122, +0.0146] | 88/88 |
| training with TTT, evaluated without | +0.0060 | [+0.0039, +0.0082] | 74/88 |
| INTERACTION | +0.0074 | [+0.0056, +0.0091] | 85/88 |

The interaction matches PG-19 at the same length (+0.0083). By domain, the gains of the
TRAINED models are largest where old context is worth most, unlike the un-tuned gain:

| domain | ceiling | TTT gain, trained with TTT | TTT gain, plain fine-tune | interaction |
|---|---|---|---|---|
| GitHub | +0.2582 | +0.0990 | +0.0752 | +0.0238 |
| arXiv | +0.2500 | +0.0425 | +0.0338 | +0.0087 |
| CommonCrawl | +0.1186 | +0.0318 | +0.0268 | +0.0050 |
| Book | +0.1092 | +0.0371 | +0.0317 | +0.0055 |

Across the 88 documents (gain = b0 + b_ceiling x ceiling + b_damage x damage; the second and
third columns add the loss level and domain indicators to see whether the ceiling coefficient
goes away):

| gain | b_ceiling | + loss level | + domain | b_damage |
|---|---|---|---|---|
| TTT at eval, trained with TTT | +0.261 [+0.206, +0.315] | +0.238 | +0.244 | +0.224 [+0.170, +0.278] |
| TTT at eval, plain fine-tune | +0.223 [+0.174, +0.273] | +0.208 | +0.215 | +0.200 [+0.153, +0.248] |
| interaction | +0.040 [+0.028, +0.053] | +0.032 | +0.032 | +0.009 [-0.004, +0.021] |

The TTT gain follows both the damage and the ceiling. The interaction follows the ceiling and
not the damage: training with TTT helps most on documents where old context is worth most.
This is a correlation across documents, not a direct measurement of memory. The recall test
below is the direct measurement.

### Recall test: does TTT remember text the model can no longer see?

`scripts/recall_probe.py`. A 1024-token passage from another book is written into a PG-19
validation sequence at tokens 2048-3071 and again at tokens 20480-21503. We measure the loss
on the second copy, with and without the first copy in the input. recall = loss without the
first copy - loss with it (nats per token; the first 32 tokens of the second copy are not
scored). At k = 1024 attention can reach back at most 16 layers x 1023 tokens = 16,368, less
than the 17,408 tokens between the copies, so a model without TTT cannot see the first copy
and its recall must be exactly 0. 32 sequences, 20 books.

| model | recall with TTT | 95% CI | books positive | recall without TTT |
|---|---|---|---|---|
| un-tuned, k = 1024 | +0.0717 | [+0.0651, +0.0783] | 20/20 | 0 (exact) |
| 40 steps, trained with TTT (arm C) | +0.1054 | [+0.1010, +0.1097] | 20/20 | 0 (exact) |
| 40 steps, plain fine-tune | +0.1018 | [+0.0976, +0.1060] | 20/20 | 0 (exact) |
| un-tuned, inner LR 2e-5 | +0.1348 | [+0.0865, +0.1830] | 19/20 | not run |
| un-tuned, inner LR 5e-5 | +0.1348 | [-0.7429, +1.0125] | 11/20 | not run |
| un-tuned, copies 4096 tokens apart | +0.0883 | [+0.0801, +0.0965] | 21/21 | -0.0007 [-0.0016, +0.0002] |
| un-tuned, FULL attention, no TTT | - | | | +2.6920 [+2.4448, +2.9393] |

Full attention copies the passage almost perfectly (loss 2.6585 without the first copy, 0.0072
with it). Trained with TTT minus plain fine-tune, same sequences: +0.0036 [+0.0018, +0.0053],
18 of 20 books. Trained with TTT minus un-tuned: +0.0337 [+0.0262, +0.0413], 20 of 20.

Prediction written before the runs: +0.01 to +0.05. Observed: +0.07 to +0.11.

1. TTT stores the passage in the fast weights. Recall is positive in every book and exactly
   zero without TTT, so nothing else can explain it.
2. The memory is weak: about 4% of what full attention recalls (0.1054 of 2.6920).
3. Training with TTT improves recall only slightly over plain fine-tuning (+0.0036, about 3%).
   Most of the improvement over the un-tuned model comes from fine-tuning itself.
4. A larger inner step stores more (+0.0717 at 4e-6, +0.1348 at 2e-5) until the model becomes
   unstable (5e-5). The step size that is best for the loss is not the best for memory.
5. Stacked attention windows pass nothing: 4096 tokens apart, recall without TTT is -0.0007.

### Recall against the inner learning rate

Same test, same 32 sequences (`scripts/della/recall_lr.sbatch`, jobs 14244517 and 14244518).
"Loss" is the ordinary loss on the 32 standard validation sequences.

| un-tuned model, inner LR | recall | 95% CI | loss |
|---|---|---|---|
| 4e-6 | +0.0717 | [+0.0651, +0.0783] | 4.5232 |
| 7e-6 | +0.1139 | [+0.1032, +0.1247] | 4.5199 |
| 1e-5 | +0.1438 | [+0.1299, +0.1577] | 4.5493 |
| 1.4e-5 | +0.1563 | [+0.1263, +0.1862] | 4.6035 |
| 2e-5 | +0.1348 | [+0.0865, +0.1830] | 4.7039 |
| 3e-5 | +0.1240 | [+0.0600, +0.1880] | 4.9499 |
| 5e-5 | +0.1348 | [-0.7429, +1.0125] | 6.9575 |

| 40-step weights trained with TTT at 4e-6, evaluated at | recall | 95% CI | loss |
|---|---|---|---|
| 4e-6 | +0.1054 | [+0.1010, +0.1097] | 2.6777 |
| 1e-5 | +0.2652 | [+0.2418, +0.2885] | 2.7088 |
| 2e-5 | +0.3880 | [+0.3105, +0.4655] | 2.8318 |
| 4e-5 | +0.4181 | [+0.2898, +0.5465] | 3.1689 |

1. On the un-tuned model recall peaks near 1.4e-5, at about twice its value at 4e-6. Going from
   4e-6 to 7e-6 costs nothing in loss and raises recall by 59%.
2. The trained weights store far more at a larger step: +0.3880 at 2e-5, 3.7 times the value at
   the rate they were trained with, and 14% of full attention (2.6920). The un-tuned model gets
   only +0.1348 at the same rate, so fine-tuning is what makes the larger step usable.
3. The larger step costs loss on ordinary text (2.6777 to 2.8318 at 2e-5). These weights were
   trained at 4e-6, so a larger step at test time is a setting they never saw. Training at the
   larger step is the obvious next run.

### More recall runs: window 8192, SlimPajama, number of fast blocks, training at a larger rate

Same test (`scripts/della/login_recall_more.sh`, `login_recall_ilr.sh`). Recall without TTT is
exactly 0 wherever the gap exceeds attention's reach, and within +-0.001 at window 8192.

| model | recall with TTT | 95% CI | documents positive |
|---|---|---|---|
| PG-19, window 8192, un-tuned (16 pairs) | +0.0283 | [+0.0247, +0.0318] | 14/14 |
| PG-19, window 8192, 10 steps trained with TTT | +0.0772 | [+0.0715, +0.0829] | 14/14 |
| PG-19, window 8192, 10 steps plain fine-tune | +0.0755 | [+0.0690, +0.0819] | 14/14 |
| SlimPajama, window 1024, un-tuned | +0.1076 | [+0.0919, +0.1233] | 30/30 |
| SlimPajama, 40 steps trained with TTT | +0.0971 | [+0.0907, +0.1034] | 30/30 |
| SlimPajama, 40 steps plain fine-tune | +0.0922 | [+0.0858, +0.0986] | 30/30 |
| SlimPajama, full attention, no TTT | +2.2611 | [+1.9826, +2.5396] | 30/30 |
| PG-19, window 1024, un-tuned, 2 fast blocks | +0.0413 | [+0.0362, +0.0464] | 20/20 |
| PG-19, window 1024, un-tuned, 4 fast blocks | +0.0717 | [+0.0651, +0.0783] | 20/20 |
| PG-19, window 1024, un-tuned, 8 fast blocks | +0.1021 | [+0.0925, +0.1118] | 20/20 |
| PG-19, window 1024, 40 steps TRAINED at inner LR 1e-5, scored at 1e-5 | +0.2931 | [+0.2770, +0.3091] | 20/20 |

1. More fast blocks store more: +0.04, +0.07, +0.10 for 2, 4 and 8 blocks at the same step per weight.
2. Training at the larger inner rate keeps the recall and removes its cost. The weights trained at
   1e-5 score 2.6820 on ordinary text (2.7337 with TTT off); the weights trained at 4e-6 score
   2.6777, and 2.7088 when only the test-time rate is raised to 1e-5. Recall is +0.2931 against
   +0.1054: 2.8 times more at +0.004 in loss.
3. Recall is flat along the passage (+0.0707, +0.0722, +0.0788, +0.0731 by quarter, un-tuned).
   A mismatch between the two copies' inputs would give a rising curve, so the limit is the
   size of the write, not the read.

### Why the write is weak (five agents, 2026-09-21), and the change that follows

Five sub-agents studied the problem from different angles. Four ran small experiments on a 135M
model or a toy built from this repo's classes; NONE of their numbers is measured on Llama. Three
of them reached the same explanation independently:

- A chunk's update to a fast matrix is G = sum_t d_t k_t^T, where k_t is the matrix input at
  token t (the key) and d_t the error at its output (the value). Reading with input q gives
  sum_t d_t (k_t . q): attention over past tokens without the softmax, in a fixed size.
- To first order, recall = step norm x gradient norm x (cosine with the passage's gradient -
  cosine with unrelated text). Their replica gives 0.9 to 1.25 nats per unit of step norm; ours
  is 0.0717 / 0.057 = 1.26. So one step of our size can store only about what we measure, and
  later updates erase little (ours: +0.0883 four chunks apart, +0.0717 seventeen apart).
- The step cannot simply grow. About 78% of ||G||^2 lies in a few key directions that all tokens
  share. That part acts like a bias on every later token and breaks the model as the step grows;
  the token-specific part, which stores "this context -> this next token", gets what is left.
- Removing or equalizing the shared directions let their small models take a 10 to 30 times
  larger step: recall rose from 3-5% to 16-27% of full attention at unchanged loss.
- Under truncated backpropagation the outer loop cannot learn to write for later reads: a read in
  a later truncation window gives the write no credit, and the learned step sizes get a gradient
  of the wrong sign (cosine -0.90 with the exact gradient in their toy). With AdamW at 4e-4 for
  40 steps the step-size multipliers could not leave [0.992, 1.008] in any case.

The TTT-E2E paper reports the same weakness (checked in its Table 2): on needle-in-a-haystack at
32K it scores 0.24, a sliding window alone 0.26, full attention 1.00, and the authors write that
their method "leaves out seemingly irrelevant details, such as the target string".

The change, inside the fast/slow framework (`--inner preconditioned_sgd`, `scripts/key_basis.py`):
normalize D = G - (1 - c) (G E) E^T instead of G, where the columns of E are the strongest
eigenvectors of E[k k^T] on training text. c = 1 is normalized SGD exactly. The planned second
stage makes this low-rank matrix a slow weight, so the outer loop learns how to write.

First Llama measurement of the related Muon update (all singular values of G set to 1): at the
SAME nominal rate it is worse than normalized SGD on the un-tuned model (first 16 pairs and
sequences: Muon 2e-5 recall +0.1263, loss 4.8869; normalized SGD 1e-5 recall +0.1393, loss 4.7142;
no TTT 5.1782). The agents predicted it needs a 10 to 30 times larger rate; those runs are in
`scripts/della/recall_muon.sbatch`.

### Two write rules that equalize the update: measured on Llama

Same recall test and the same 32 sequences as above (`scripts/della/login_recall_preconditioned.sh`,
`scripts/della/recall_muon.sbatch`). "Loss" is the ordinary loss on the 32 standard validation
sequences. The two rules:

- **preconditioned**: the 64 strongest shared key directions of each fast matrix are removed
  from the gradient before it is normalized (`--inner preconditioned_sgd`, c = 0). On Llama one
  direction carries 32-60% of the key energy of a fast matrix and 64 directions carry 55-83%
  (`scripts/key_basis.py`, 65,536 training tokens).
- **Muon**: every singular value of the gradient is set to 1 before the step (`--inner muon`).
  The largest singular value of a normalized-SGD update is up to sqrt(min(m, n)) = 45 times
  Muon's at the same per-element size, which is why Muon tolerates a much larger rate.

Un-tuned model (best normalized-SGD entries repeated for comparison):

| rule | inner LR | recall | 95% CI | loss |
|---|---|---|---|---|
| no TTT | - | 0 | | 4.9895 |
| normalized SGD | 4e-6 | +0.0717 | [+0.0651, +0.0783] | 4.5232 |
| normalized SGD | 1.4e-5 | +0.1563 | [+0.1263, +0.1862] | 4.6035 |
| preconditioned | 4e-6 | +0.0739 | [+0.0685, +0.0793] | 4.8059 |
| preconditioned | 1e-5 | +0.1748 | [+0.1645, +0.1850] | 4.6634 |
| preconditioned | 2e-5 | +0.3013 | [+0.2790, +0.3236] | 4.5698 |
| preconditioned | 4e-5 | +0.4858 | [+0.4368, +0.5348] | 4.5517 |
| preconditioned | 1e-4 | unstable | | 7.2559 |
| preconditioned, 30% of the shared part kept | 2e-5 | +0.2533 | [+0.2134, +0.2933] | 4.6275 |
| Muon (16 pairs, 16 sequences) | 2e-5 | +0.1263 | [+0.1198, +0.1360] | 4.8869 |
| Muon | 4e-5 | +0.2488 | [+0.2366, +0.2609] | 4.5619 |
| Muon | 1.2e-4 | +0.6865 | [+0.6488, +0.7242] | 4.3290 |

40-step weights trained with normalized SGD at 4e-6 (the preconditioned rule uses key directions
measured on these weights; they are less concentrated: 47-73% in 64 directions):

| rule | inner LR | recall | 95% CI | loss |
|---|---|---|---|---|
| normalized SGD | 4e-6 | +0.1054 | [+0.1010, +0.1097] | 2.6777 |
| normalized SGD | 2e-5 | +0.3880 | [+0.3105, +0.4655] | 2.8318 |
| preconditioned | 1e-5 | +0.2423 | [+0.2271, +0.2575] | 2.6746 |
| preconditioned | 2e-5 | +0.4513 | [+0.4137, +0.4889] | 2.6907 |
| preconditioned | 4e-5 | +0.7025 | [+0.5986, +0.8064] | 2.7648 |
| preconditioned | 1e-4 | unstable | | 5.4698 |
| Muon | 2e-5 | +0.2066 | [+0.1937, +0.2195] | 2.6742 |
| Muon | 4e-5 | +0.4183 | [+0.3918, +0.4448] | 2.6636 |
| Muon | 1.2e-4 | +1.0241 | [+0.9555, +1.0927] | 2.6895 |

1. Both rules do what the diagnosis predicts: at equal loss they store several times more.
   At a step 30 times the old one, Muon on the trained weights recalls +1.0241, 38% of full
   attention (2.6920), against 4% before, and the loss is within 0.012 of the old value. On the
   un-tuned model Muon at 1.2e-4 gives 9.6 times the recall AND a lower loss (4.3290 against
   4.5232).
2. Muon beats the preconditioned rule at every step size tried, and stays stable to a larger
   step. Removing 64 key directions leaves a gradient whose remaining directions are still
   uneven; Muon flattens all of them.
3. On the un-tuned model at the OLD step size the preconditioned rule hurts the loss (4.8059):
   the shared directions are part of what repairs the broken window there. At larger steps this
   no longer shows.
4. Cost: Muon's Newton-Schulz iteration runs in float32 and made an evaluation pass 20 seconds
   per sequence on an A100 against 3.4; the preconditioned rule costs nothing measurable. The
   iteration can run in bf16 or under TF32 (not done yet). Nothing is meta-trained through
   either rule yet; short validation runs of Muon meta-training were submitted (jobs 14247911,
   14247912) to measure its memory and speed.

### PG-19 at 128K, nothing trained

Books of at least 131,073 tokens, so no sequence spans two books; every 25th book held out:
6,505 training books (1,466,573,365 tokens), 272 validation books (66,629,305 tokens).
k = 8192, b = 1024, 16 validation sequences from 16 books (`scripts/della/eval_128k.sbatch`).

| run | loss |
|---|---|
| arm A (no TTT) | 4.2900 |
| arm B (normalized SGD, 4e-6) | 3.9290 |

TTT alone: +0.3610 nats per book, 95% CI [+0.2996, +0.4225], 16 of 16 books
(`scripts/paired_ttt_effect.py --baseline`). By position:

| tokens | TTT on | TTT off | difference |
|---|---|---|---|
| 0 - 8192 (inside the window) | 2.5445 | 2.5476 | +0.0031 |
| 8192 - 16384 | 4.0942 | 4.2119 | +0.1177 |
| 24576 - 32768 | 4.1401 | 4.4391 | +0.2991 |
| 57344 - 65536 | 3.9811 | 4.3631 | +0.3820 |
| 90112 - 98304 | 4.0219 | 4.4590 | +0.4371 |
| 122880 - 131072 | 3.8357 | 4.3581 | +0.5224 |

The 32K signature, stronger: nothing inside the window, then a gain that keeps growing with
position, to +0.52 nats in the last band. With TTT the loss beyond the window FALLS along the
sequence (4.14 to 3.84); without it, it stays between 4.21 and 4.52. Both arms are far above the
2.54 seen inside the window, so this is the repair regime again, and at 32K at least 85% of
such a gain was repair. What out-of-window context is worth at 128K is scored in pieces
(`scripts/della/cv_128k.sbatch`): one full-attention float32 forward at 128K takes about
5 minutes on an A100, and the first attempt, inside the evaluation job, had scored 5 of 12
sequences after 31 minutes and was cancelled with nothing saved (job 14238061).

First piece (6 sequences, 6 books; job 14239478, 33 minutes), S = 8192, tightest band:
full 2.6419, restart 2.6742, value +0.0323 per book, 95% CI [-0.0000, +0.0647], 6 of 6 positive;
sanity difference exactly 0. At 128K the ceiling on PG-19 is about +0.03 against +0.0208 at 32K:
four times the context adds little that a healthy model can use. Books are not where a
long-range memory pays. Second piece (6 more books, job 14239517): +0.0385 [+0.0197, +0.0573],
6 of 6 positive.

### The matched budget: corpus and jobs

The 32K PG-19 training split held 300,119,276 tokens; one pass of the matched budget consumes
760,217,600 (725 x 32 x 32768), and the loader cycles silently when the corpus runs out. The
orchestrator's gate refused to submit. `scripts/della/extend_pg19.sbatch` built `pg19_32k_full`
from the whole PG-19 training stream under the same split rule: 7,626 training books,
900,080,292 tokens. Verified before use: the new validation file begins with the old one byte
for byte (34,832,344 bytes) and the new training file with the old one (1,200,477,104 bytes);
the old validation split was then installed, so every evaluation scores the same 64 books as
before.

Queued on 2026-09-21, all pending on priority; each is a chain of two links, the second a spare
that resumes from the checkpoint or exits at once:

| jobs | run | resources per link |
|---|---|---|
| 14237614, 14237615 | arm C, k = 8192, 725 steps x 32 sequences, `--eval-ttt-off` | 4 GPUs, 60 h |
| 14237616, 14237617 | its `--inner none` control | 4 GPUs, 34 h |
| 14237645, 14237646 | arm C, k = 1024, `truncate_bptt=4`, same budget | 4 GPUs, 24 h |
| 14237647, 14237648 | its `--inner none` control | 4 GPUs, 16 h |

The single-GPU runs queued on 2026-09-20 (`C32k_t1`, `C32k_t2`, `C32k_ctl20`, `C32k_bs8`,
`C32k_bs16`, `C32k_bs32`, `C32k_s60`, `C32k_s150`, `C32k_bs32s60`) are still pending.

## 2026-09-23: the window-8192 ladder, and two more write-rule numbers

### Longer training and larger batches at the reference window

The single-GPU jobs queued on 2026-09-20 ran on 2026-09-23. PG-19, T = 32768, window 8192,
normalized SGD 4e-6, outer lr 4e-4, LoRA r = 64, `truncate_bptt=2` unless noted, evaluated on
the standard 32 sequences (22 books). "TTT off" is the same trained weights with the inner loop
switched off; the per-book column is that difference clustered by book.

| run | steps | sequences per step | tokens | loss, TTT on | TTT off | TTT on - off per book | 95% CI | books positive |
|---|---|---|---|---|---|---|---|---|
| `C32k_t2` | 20 | 4 | 2.6M | 2.5958 | not evaluated | | | |
| `C32k_t1` (`truncate_bptt=1`) | 20 | 4 | 2.6M | 2.5966 | not evaluated | | | |
| `C32k_ctl20` (plain fine-tune, `--inner-lr 0`) | 20 | 4 | 2.6M | 2.6139 | | | | |
| `C32k_bs8` | 20 | 8 | 5.2M | 2.5799 | 2.5943 | | | |
| `C32k_bs16` | 20 | 16 | 10.5M | 2.5738 | 2.5909 | | | |
| `C32k_bs32` | 20 | 32 | 21.0M | 2.5664 | 2.5777 | +0.0091 | [+0.0047, +0.0134] | 22/22 |
| `C32k_s60` | 60 | 4 | 7.9M | 2.5146 | 2.5294 | +0.0125 | [+0.0082, +0.0167] | 22/22 |
| `C32k_s150` | 150 | 4 | 19.7M | 2.4664 | 2.4756 | +0.0067 | [+0.0025, +0.0110] | 21/22 |

Paired comparisons (`scripts/paired_ttt_effect.py --baseline`):

- Trained through the inner loop against plain fine-tuning at 20 steps (`C32k_t2` against
  `C32k_ctl20`): +0.0186 per book [+0.0146, +0.0226], 21 of 22. Inside the window the plain
  fine-tune is better (-0.0115); in the last 8192 tokens the meta-trained model is better
  (+0.0338).
- Truncating the meta-gradient to 1 chunk instead of 2 changes nothing at 20 steps: -0.0004
  per book [-0.0014, +0.0006], 11 of 22.

1. What TTT adds on the same weights shrinks as training goes on: +0.0248 (8 steps, from
   "Same weights, inner loop on and off"), +0.0125 (60 steps), +0.0067 (150 steps). The
   ceiling at this window is +0.0208. At 150 steps the model is still 0.14 nats above the
   healthy windowed level (about 2.33), so the repair is not finished either.
2. At equal tokens, more steps beat a larger batch: 150 steps of 4 sequences (19.7M tokens)
   reach 2.4664; 20 steps of 32 sequences (21.0M tokens) reach 2.5664.
3. Plain controls for the 60- and 150-step runs were submitted on 2026-09-23 (jobs 14330258
   and 14330259). `C32k_bs32s60` (60 steps of 32 sequences) is running.

### The 2x2 at window 8192, 20 steps

Both 20-step weight sets evaluated with TTT on and off (`scripts/della/login_cells_k8192_s20.sh`);
the loaded weights reproduced the training jobs' own numbers (2.5957 against 2.5958; 2.6139).

| slow weights | TTT on at eval | TTT off at eval |
|---|---|---|
| trained through the inner loop | 2.5957 | 2.6133 |
| plain fine-tune | 2.6040 | 2.6139 |

| effect, per book (22 books) | 10 steps (from "The 2x2 behind H1") | 20 steps | 95% CI at 20 steps | books positive |
|---|---|---|---|---|
| TTT at eval, weights trained with TTT | +0.0179 | +0.0142 | [+0.0080, +0.0205] | 22/22 |
| TTT at eval, plain fine-tuned weights | +0.0164 | +0.0084 | [+0.0047, +0.0121] | 21/22 |
| training with TTT, evaluated with TTT | +0.0247 | +0.0104 | [+0.0059, +0.0149] | 21/22 |
| training with TTT, evaluated without | +0.0232 | +0.0045 | [-0.0028, +0.0119] | 21/22 |
| INTERACTION | +0.0015 | +0.0058 | [+0.0028, +0.0089] | 22/22 |

At the reference window the interaction grew from 10 to 20 steps (+0.0015 to +0.0058) while
every other effect shrank, and it is now positive in all 22 books. The training effect with
TTT off has fallen to +0.0045 and is no longer clearly different from zero: the head start on
window repair that training through the inner loop gave at 10 steps is gone by 20. What TTT
is worth on the plain fine-tuned weights halved (+0.0164 to +0.0084); on the meta-trained
weights it fell less (+0.0179 to +0.0142). The 60- and 150-step 2x2 follow when their
controls finish.

### Where Muon breaks, and training at 2e-5

Same recall test and sequences as "Two write rules that equalize the update".

| weights | inner rule, rate | recall | 95% CI | loss |
|---|---|---|---|---|
| 40 steps at 4e-6, tested at | Muon 2.4e-4 | +1.5005 | [+1.3894, +1.6117] | 2.7937 |
| 40 steps TRAINED at normalized SGD 2e-5, tested at 2e-5 | normalized SGD 2e-5 | +0.5201 | [+0.4733, +0.5669] | 2.7243 |

Muon at 2.4e-4 (60 times the old step) is still stable and recalls 56% of full attention, at a
loss 0.116 above the 4e-6 value. Training at 2e-5 (rather than only testing at it) raises
recall from +0.3880 to +0.5201 and lowers the loss from 2.8318 to 2.7243 (TTT off: 2.7847),
but the loss stays 0.047 above the weights trained at 4e-6. The first validation of
meta-training THROUGH Muon (jobs 14247911, 14247912) failed before training: `--steps 3` with
the 10% warmup rounds to a 0-step warmup, which is a hard error by design. Resubmitted with
`--steps 5` (jobs 14330212, 14330213), which fails the same way (Python rounds 0.5 to 0; the
message had named the wrong fix, now corrected), then with `--steps 6`.

### Meta-training through Muon fits, and its cost

Job 14330856 (window 1024, `truncate_bptt=4`, Muon at 1.2e-4, 6 steps of 4 sequences, A100):
286 s per step, peak 60.4 GiB; loss 5.0223 at step 0 and 3.3282 at step 5; evaluation on 4
sequences 3.2628 (TTT off 3.3774). At `truncate_bptt=2` (job 14330857): 250 s per step. For
comparison normalized SGD takes 55 s per step at truncation 4, so the second-order pass through
the Newton-Schulz iteration costs about 5x. The 40-step run was submitted as a chain of five
1-hour `gpu-test` links (jobs 14333213 to 14333217) with the existing plain control
`C_32k_k1024_ctl_s40` as its 2x2 partner.

The 60-step plain control at window 8192 (`C32k_ctl60`, job 14330258, 2 GPUs, 37 minutes):
2.5255, against 2.5146 (TTT on) and 2.5294 (TTT off) for the weights trained through the inner
loop (`C32k_s60`). Its evaluation with TTT on, and the same for the 150-step pair, run from
`scripts/della/login_cells_k8192_s60_s150.sh`.

### The 2x2 at window 8192, 60 steps

The 60-step plain weights (`C32k_ctl60`) evaluated with TTT on and off on the login GPU
(`scripts/della/login_cells_k8192_s60_s150.sh`, `results/cell_plainft_60.json`); the TTT-off
evaluation reproduced the control's own 2.5255. Same settings as the 20-step 2x2.

| slow weights | TTT on at eval | TTT off at eval |
|---|---|---|
| trained through the inner loop (`C32k_s60`) | 2.5146 | 2.5294 |
| plain fine-tune (`C32k_ctl60`) | 2.5187 | 2.5255 |

| effect, per book (22 books) | 20 steps | 60 steps | 95% CI at 60 steps | books positive |
|---|---|---|---|---|
| TTT at eval, weights trained with TTT | +0.0142 | +0.0125 | [+0.0082, +0.0167] | 22/22 |
| TTT at eval, plain fine-tuned weights | +0.0084 | +0.0048 | [+0.0008, +0.0087] | 16/22 |
| training with TTT, evaluated with TTT | +0.0104 | +0.0053 | [+0.0024, +0.0081] | 21/22 |
| training with TTT, evaluated without | +0.0045 | -0.0024 | [-0.0064, +0.0015] | 5/22 |
| INTERACTION | +0.0058 | +0.0077 | [+0.0058, +0.0096] | 21/22 |

At the reference window the interaction has grown at every step count so far: +0.0015 (10
steps), +0.0058 (20), +0.0077 (60). With TTT off, the plain fine-tune is now slightly better
than the weights trained through the inner loop (-0.0024, 17 of 22 books), so training through
the inner loop no longer gives better slow weights on its own; what it gives is weights that
use TTT better. On the plain weights TTT is worth only +0.0048 and is positive in 16 of 22
books. Prediction for the 150-step 2x2, written before its control finished: interaction
between +0.005 and +0.010; TTT on the plain weights below +0.005.

## 2026-09-24: meta-training through Muon, and what it does not buy

### The 40-step run through Muon

Arm C at window 1024, `truncate_bptt=4`, 4 sequences per step, outer lr 4e-4, 40 steps, with
Muon at 1.2e-4 as the inner rule DURING training (jobs 14333213 to 14333217, five 1-hour links
on `gpu-test`; 301 s per step on an A100 against 55 for normalized SGD; peak 60.7 GiB). The
learned step multipliers ended at 0.995 (min 0.994, max 0.996). Predictions written before the
run (`.agent/plan.md`): loss at or below 2.6777 with TTT on; recall at or above +1.0241;
interaction at least twice +0.0083.

| weights (all scored with Muon 1.2e-4 at evaluation) | loss, TTT on | TTT off | recall | 95% CI | books positive |
|---|---|---|---|---|---|
| 40 steps trained THROUGH Muon 1.2e-4 (`C_32k_k1024_muon_s40`) | 2.6762 | 2.7380 | +1.0006 | [+0.9354, +1.0658] | 20/20 |
| 40 steps trained through normalized SGD 4e-6 (`C_32k_k1024_t4_s40`, from "Two write rules") | 2.6895 | | +1.0241 | [+0.9555, +1.0927] | 20/20 |

Paired on the same 32 planted passages, clustered by the 20 carrier books: trained through
Muon minus trained through normalized SGD, recall -0.0235 [-0.0295, -0.0175], 2 of 20 books
in favour of the Muon-trained weights.

1. The loss prediction held by 0.0015 (2.6762 against the bound 2.6777). Training through
   Muon lowers the loss under Muon by 0.013 against weights trained through normalized SGD.
2. The recall prediction failed. Meta-training through the strong write rule does not
   raise what the write stores; it lowers it slightly, in 18 of 20 books. What TTT is worth
   on these weights, +0.0618 (2.7380 against 2.6762), is twice the +0.0309 of the weights
   trained through normalized SGD, but that is the loss cost of switching a write off that the
   slow weights were trained to expect, not more memory.
3. By the rule written before the run, the recall half of "meta-learning adds nothing to a
   strong fixed write rule" is met. The interaction half is below.

### The 2x2 under Muon, 40 steps, window 1024

The plain 40-step control (`C_32k_k1024_ctl_s40`, trained with `--inner none`) scored with Muon
1.2e-4 on and off on `gpu-test` (job 14354267, file `C_32k_k1024_ctl_s40_muon_evallr1.2e-4`);
its TTT-off loss reproduced the control's own 2.7196. `scripts/two_by_two.py`, 32 sequences,
22 books. Prediction written before the cell landed: interaction +0.02 to +0.04.

| slow weights | TTT on at eval (Muon 1.2e-4) | TTT off |
|---|---|---|
| trained through the inner loop (Muon 1.2e-4) | 2.6762 | 2.7380 |
| plain fine-tune | 2.7079 | 2.7196 |

| effect, per book (22 books) | normalized SGD 4e-6, 40 steps (from 2026-09-21) | Muon 1.2e-4, 40 steps | 95% CI under Muon | books positive |
|---|---|---|---|---|
| TTT at eval, weights trained with TTT | +0.0278 | +0.0590 | [+0.0474, +0.0705] | 22/22 |
| TTT at eval, plain fine-tuned weights | +0.0195 | +0.0098 | [-0.0024, +0.0219] | 12/22 |
| training with TTT, evaluated with TTT | +0.0179 | +0.0291 | [+0.0245, +0.0336] | 22/22 |
| training with TTT, evaluated without | +0.0096 | -0.0201 | [-0.0239, -0.0164] | 1/22 |
| INTERACTION | +0.0083 | +0.0492 | [+0.0457, +0.0527] | 22/22 |

1. The interaction is six times its normalized-SGD value and above the predicted range. Read
   with its parts, it is mostly dependence, not memory: the weights trained through Muon are
   WORSE than the plain weights when the write is switched off (-0.0201, 21 of 22 books the
   other way), and better by +0.0291 when it is on. A strong write during training makes the
   slow weights rely on it; the loss they reach with it is 0.029 below what plain fine-tuning
   reaches under the same write, on a model still above the healthy level for this window
   (about 2.39).
2. On plain fine-tuned weights Muon at 1.2e-4 is worth only +0.0098 and is not clearly
   different from zero (12 of 22 books), against +0.0195 for normalized SGD at 4e-6 on the same
   weights: the large step that stores most is not the step that repairs the window best, and
   it takes the slow weights to make it pay in loss.
3. Together with the recall result above: training through Muon teaches the slow weights to
   USE a strong write for the next-token loss (+0.0291 over plain, 22/22) without making the
   write STORE more verbatim text (recall -0.0235 against the normalized-SGD-trained weights).
   On the field's headline metric, next-token loss (TTT-E2E Fig. 1, LaCT Fig. 5, TTT Fig. 2),
   the thesis is supported at this budget. On verbatim recall it is not, which is also what
   TTT-E2E reports for its own method (Table 2, "compression leaves out seemingly irrelevant
   details"). The recall test is one probe (copying, one gap, one depth) and a lower bound on
   memory, not a general memory score; see `.agent/literature.md`, "How the field measures".

The Muon recall jobs did not repeat the no-TTT floor check (`exact_floor_checked` is false in
their files); the floor is a property of attention's reach, not of the write rule, and was
exact in every normalized-SGD run on the same pairs.

### Memory at chunk 2048

`scripts/memory_probe.py --chunk 2048 --window 2048 --prefix-segment 2048 --truncate-bptt 2
--remat-blocks` (one sequence, no optimizer step, login GPU): peak 49.73 GiB, prefix 4.94 GiB
resident. At window 8192 the trainer's peak was 18 GiB above the probe's, at window 1024 about
0.6 GiB above; a validation job must measure the trainer's peak before any 40-step run.

### An unused gradient buffer

Every window's backward left d loss / d W_0 on the live fast parameters, which nothing read
or zeroed: 201M floats, 0.8 GiB, resident for the whole run in every arm C job so far
(commit d6dabb6 drops it after each sequence unless the fast init is trained). Numerics are
unchanged; the saving is not yet measured on the cluster.

### bf16 Newton-Schulz: same numbers, 2 to 4 times faster

The 40-step weights trained through normalized SGD (`C_32k_k1024_t4_s40`), Muon at 1.2e-4,
`--ns-dtype bfloat16` against the float32 rounds, same 32 pairs and 32 sequences (jobs 14354268
and 14353810, both A100). Prediction written before: recall within 0.02 of +1.0241, evaluation
under 8 s per sequence.

| Newton-Schulz dtype | recall | 95% CI | recall test, seconds | loss, TTT on | TTT off | 32-sequence loss run |
|---|---|---|---|---|---|---|
| float32 | +1.0241 | [+0.9555, +1.0927] | 1277 | 2.6895 | 2.7086 | 671 s |
| bfloat16 | +1.0239 | [+0.9554, +1.0924] | 329 | 2.6895 | 2.7086 | 336 s |

Paired per book, bf16 minus fp32 recall: -0.0003 [-0.0007, +0.0002], 7 of 20 books. The loss
agrees to four decimals. The recall test runs 3.9 times faster and the loss evaluation 2.0
times (10.5 s per sequence, not the predicted 8). Adopted: every Muon evaluation from here
passes `--ns-dtype bfloat16`; the meta-training step time under bf16 is not yet measured.

### Row reset at 1.2e-4: no change, as a small step predicts

Same weights, Muon 1.2e-4 in float32, `--weight-norm row_reset` (job 14354269): recall
+1.0236 [+0.9550, +1.0921]; paired against the plain rows, -0.0006 [-0.0010, -0.0002], 4 of
20 books; loss 2.6893 with TTT on (2.6895 without the reset), 2.7086 off. A step of per-element RMS 1.2e-4 over 32 chunks barely changes a row's norm, so the
reset has nothing to undo. Its purpose is a larger stable step. Prediction, written before
jobs 14356889 (plain rows) and 14356890 (row reset) at Muon 4.8e-4 (both bf16): without the reset the loss rises above 2.9 or the
run becomes unstable; with the reset the loss stays under 2.85 and recall exceeds +1.5005
(the 2.4e-4 value).

### Muon at 4.8e-4, with and without the row reset: the reset does nothing

Same 40-step weights (`C_32k_k1024_t4_s40`), Muon at 4.8e-4 in bf16, jobs 14356889 (plain rows)
and 14356890 (`--weight-norm row_reset`), 32 pairs and 32 sequences. Prediction on record:
without the reset the loss rises above 2.9; with it the loss stays under 2.85 and recall passes
+1.5005.

| Muon rate | row reset | recall | 95% CI | loss, TTT on | TTT off | share of full attention (+2.6920) |
|---|---|---|---|---|---|---|
| 1.2e-4 | no | +1.0241 | [+0.9555, +1.0927] | 2.6895 | 2.7086 | 38% |
| 2.4e-4 | no | +1.5005 | [+1.3894, +1.6117] | 2.7937 | | 56% |
| 4.8e-4 | no | +1.9159 | [+1.7622, +2.0697] | 3.0559 | 2.7086 | 71% |
| 4.8e-4 | yes | +1.9152 | [+1.7616, +2.0688] | 3.0542 | 2.7086 | 71% |

1. The first half of the prediction held (3.0559 is above 2.9) and the second failed: the row
   reset changes neither recall nor loss at 4.8e-4, as it did not at 1.2e-4. Forty steps of
   per-element RMS 4.8e-4 do not move a pretrained row's norm enough for a reset to matter;
   the loss cost of a large write is not norm drift. The option stays in the code as an
   explicit opt-in and leaves the plan.
2. Recall keeps rising with the step, to 71% of full attention, while the loss cost grows from
   +0.012 (1.2e-4) to +0.104 (2.4e-4) to +0.366 (4.8e-4) on weights that never saw these steps
   in training. Training at 2e-5 removed most of the loss cost of that step for normalized SGD
   (2.7243 against 2.8318); the same test for Muon at 2.4e-4 is the next chain.

### Six-step validations: per-token rates and arm F fit and run at the plain step time

Window 1024, `truncate_bptt=4`, normalized SGD 4e-6, 6 steps of 4 sequences, 4 evaluation
sequences, one A100 each (jobs 14356891, 14356892).

| run | fast | slow | outer | step 0 loss | s per step | peak | TTT on / off (4 sequences) |
|---|---|---|---|---|---|---|---|
| arm C `--token-rates` | 201,326,592 | 45,164,560 | 45,164,560 | 5.066569 | 54.7 | 42.2 GiB | 3.2936 / 3.3638 |
| arm C plain (`C_32k_k1024_t4_s40`, same first step) | 201,326,592 | 45,156,364 | 45,156,364 | 5.066569 | 55 | 42.0 GiB | |
| arm F `--prime-intermediate 2048` | 50,331,648 | 45,172,752 | 95,504,400 | 5.218563 | 53.0 | 36.2 GiB | 3.4729 / 3.4737 |

1. The token-rate run's step-0 loss equals the plain run's to every printed digit, as
   predicted (eta = 1 at init); the rates cost no time; after 6 steps the rate weights have
   norm 0.03 per block (checkpoint read), so they are learning.
2. Arm F's parameter counts are the designed ones (3 x 2048 x 2048 x 4 fast; 4 gates and 8
   norm gains added to the slow set; outer = slow + fast). Its gates left zero but only to
   +-0.001 after 6 steps, so the prime MLP contributes nothing yet (TTT on against off
   +0.0008). At the outer rate of 4e-4 under AdamW a gate moves at most about 4e-4 per step,
   so 40 steps reach about 0.016. The 40-step pair will show whether that is enough; if not,
   the gate needs its own learning rate or a small nonzero init, a deliberate deviation from
   LaCT's zero init that would be recorded as such.
3. Submitted on `gpu-test` at 03:20: `C_32k_k1024_tokrates_s40` (against `C_32k_k1024_t4_s40`,
   same data and schedule, then recall), `F_32k_k1024_s40` and its `--inner none` control
   `F_32k_k1024_ctl_s40` (jobs 14357823 to 14357825). Predictions: token rates lower the
   40-step loss against the plain run by 0.002 to 0.01 per book and raise recall by under
   0.02; arm F's gates end near 0.016 and its TTT effect is under +0.005.

### Forty steps with per-token rates: no change in loss

`C_32k_k1024_tokrates_s40` (job 14357823): arm C at window 1024 with `--token-rates`, otherwise
the settings and data order of `C_32k_k1024_t4_s40`; 54.2 s per step, peak 42.2 GiB. Prediction
on record: 0.002 to 0.01 lower loss per book. Observed: TTT on 2.6780 against 2.6777, TTT off
2.7077 against 2.7086; paired per book -0.0002 [-0.0010, +0.0005], 9 of 22 books. The rate
weights reached norm 0.03 per block and the learned step multipliers spread to [0.994, 1.007].
The prediction failed: under normalized SGD at 4e-6 the learned per-token weighting does not
change the loss. Its recall is pending (job 14359038, Muon 1.2e-4 in bf16 with the rates on).

### Forty steps of arm F: the zero gate never opens

`F_32k_k1024_s40` and its `--inner none` control `F_32k_k1024_ctl_s40` (jobs 14357824,
14357825): prime MLP of width 2048, output RMSNorm, gate at 0, normalized SGD 4e-6 on the prime
weights, the prime W_0 trained by the outer loop; 53.5 and 29.3 s per step; peak 36.2 GiB.

| arm F, 40 steps | TTT on | TTT off | gate mean at step 39 |
|---|---|---|---|
| trained through the inner loop | 2.7161 | 2.7176 | -0.00015 (min -0.0007) |
| plain (`--inner none`) | | 2.7188 | -0.00010 |

Paired per book: TTT on against off +0.0013 [+0.0009, +0.0018], 22/22; against the control
+0.0023 [+0.0014, +0.0031], 22/22. Both are real and both are tiny: with the gate at 1e-4 the
prime MLP is silent and the numbers are the LoRA's. The gate does not open because the prime
MLP starts as noise, so opening it raises the loss, while the prime MLP gets no gradient
until it opens. LaCT trains through this from scratch over tens of billions of tokens; forty
steps of four sequences cannot. Deviation recorded: `--prime-gate-init` (commit on
2026-09-24) starts the gates at a chosen value; the next pair uses 0.1. Prediction: with the
gate at 0.1 the prime W_0 moves, arm F's TTT effect exceeds +0.005, and its loss stays within
0.01 of the plain arm C level (2.6979 with the write on).

### bf16 Newton-Schulz during meta-training: adopted

6-step validation (job 14357884; Muon 1.2e-4, window 1024, `truncate_bptt=4`) against the fp32
validation (job 14330856): step-0 loss 5.022287 against 5.022278, step-5 loss 3.328198 against
3.328211, evaluation on 4 sequences 3.2629 against 3.2628; 81.6 s per step against 286 (3.5x);
peak 52.0 against 60.4 GiB. All three predictions held. The 40-step chain through Muon at 2.4e-4
in bf16 is submitted as two 1-hour links (jobs 14359068, 14359069); predictions in
`.agent/plan.md`.

### Sixty steps of 32 sequences at window 8192: three times the tokens of the 150-step run, the same loss

`C_32k_bs32_s60` (job 14169729; 60 x 32 sequences, 62.9M tokens, one A100, 1078 s per step,
18.2 h, peak 68.0 GiB; normalized SGD 4e-6, `truncate_bptt=2`).

| run | steps x sequences | tokens | loss, TTT on | TTT off | TTT on - off per book | 95% CI | books |
|---|---|---|---|---|---|---|---|
| `C32k_bs32` | 20 x 32 | 21.0M | 2.5664 | 2.5777 | +0.0091 | [+0.0047, +0.0134] | 22/22 |
| `C32k_s150` | 150 x 4 | 19.7M | 2.4664 | 2.4756 | +0.0067 | [+0.0025, +0.0110] | 21/22 |
| `C32k_bs32_s60` | 60 x 32 | 62.9M | 2.4687 | 2.4768 | +0.0063 | [+0.0030, +0.0095] | 21/22 |

Paired per book, `bs32_s60` against `s150` with TTT on: -0.0066 [-0.0154, +0.0023], 1 of 22
books in favour of the larger batch (one book carries the wide interval); against `bs32`,
+0.0888 [+0.0715, +0.1061], 22/22. Three times the tokens in batches of 32 buy the same loss as
150 steps of 4: at this stage the number of outer steps matters, not the number of tokens, and
what TTT adds on the same weights stays at +0.006 to +0.007, under the +0.0208 ceiling. It has
no plain control of its own.
