## Research Question Brief

**Mode**: deep-research `lit-review` + Phase 1 scoping (RQ Brief + DA Checkpoint 1)
**Date**: 2026-09-16
**Inputs**: Tandon, Dalal, Li, Koceja, Rød et al. (2025), "End-to-End Test-Time Training for Long Context", arXiv:2512.23675v2; code https://github.com/test-time-training/e2e (commit a4fc478); user design brief (chat).

### Topic Area
Inverting the size relationship between fast (inner-loop, test-time-updated) and slow (outer-loop, meta-learned) weights in TTT-E2E: make the fast weights large (full MLP weight matrices of many/all Transformer blocks) and make the slow weights a small LoRA adapter, while keeping the paper's SWA + chunked (mini-batch) TTT design and the DCLM evaluation protocol.

### Baseline facts established from paper + code (ground truth for all downstream phases)
| Item | TTT-E2E (paper/code) |
|---|---|
| Architecture | Llama-style Transformer, SwiGLU MLP, RMSNorm pre+post norm, QK-norm, RoPE θ=500K, tied embeddings, Llama-3 tokenizer |
| Sequence mixer | Sliding-window attention, window k=8192 (`SWA` class keeps a KV cache of k tokens in `eqx.nn.State`; each 1024-token chunk attends to [cache ∥ chunk] with a sliding causal mask). Prefix (non-TTT) blocks run once over the full sequence with cuDNN flash local attention. |
| Fast weights (inner loop) | Only `feed_forward_prime` (a second SwiGLU MLP) in the last 1/4 of blocks (`suffix_len` = 3/12, 6/24, 8/32). Intermediate size shrunk (2048→1664 at 125M) so total params match baseline. Norms, attention, embeddings frozen in inner loop. Fast-weight size: 11.5M (125M), 160M (1B), 346M (3B) = ~9-11% of params. |
| Slow weights (outer loop) | **All** parameters incl. the fast-weight initialization W0 (`spec_outer: ["**"]`). |
| Chunk / mini-batch size b | 1024 tokens (`mini_batch_size`); ablation shows b>1K hurts, b<1K too slow/unstable. Constraint k ≥ b. |
| Inner optimizer | `optax.sgd`, lr=1.0, `clip_by_global_norm(1.0)` over all fast params jointly → effectively **normalized SGD with step norm 1** once ‖g‖>1. No momentum, no state. Inner-LR warmup 0.1→1.0 over first 10% of pretraining (`ilr_warmup_steps`), none for extension. |
| Inner loss | Next-token CE on chunk i with W_{i-1}; then one gradient step → W_i. Outer loss = mean over chunks of loss-before-update (Eq. 6). |
| Outer optimizer | AdamW b1=0.9 b2=0.95 wd=0.1 clip=1.0; peak LR 3e-3 (125M) … 8e-4 (3B) pretrain, 4e-4 extension; 10% linear warmup then cosine to 1e-5. |
| Gradient-of-gradient | `eqx.filter_value_and_grad` inside `jax.lax.scan`; outer `filter_value_and_grad` differentiates through the whole scan (second-order, MAML-style). |
| Memory strategy | Checkpointing **through time**: each chunk step wrapped in `jax.remat` (policy `nothing_saveable`) so only the carry (fast weights W_i in fp32 `state_dtype`, inner opt state, KV-cache state) is saved per chunk boundary; `scan_remat_chunk(inner_remat_freq)` adds a second level (remat groups of n chunks) → ~2·sqrt(T/b) copies. Per-device batch handled by `vmap` (memory × batch) or `accum_steps` (sequential `lax.scan`, no memory growth). Fast-weight sharding across GPUs via `n_state_parallel` mesh axis. |
| Data / eval | Pretrain on DCLM-Baseline docs ≥8K tokens at 8K context (Chinchilla tokens: 2.5B @125M, 15B @760M); extension fine-tune on Books at 32K-128K (5% of pretrain tokens, batch doubled). Metric: held-out loss (log-ppl) and per-token-index loss curve (`token_nll_loss`). |
| Attention kernels | cuDNN flash used only where no second-order grads needed (prefix blocks); TTT suffix uses XLA attention because flash kernels lack double backward. |

### Primary Research Question
Under the TTT-E2E protocol (SWA window 8K, chunked inner-loop TTT, DCLM 8K pretraining data and held-out loss), does a configuration with **large fast weights (full MLP matrices of ≥1/4 to all blocks, initialized from a pretrained SWA Transformer) and small slow weights (a meta-learned LoRA adapter)** match or improve held-out loss and per-token loss decay relative to (a) the frozen SWA baseline, (b) TTT-naive (same inner loop, no meta-learning), and (c) the paper's TTT-E2E (small fast, all-slow), at matched total parameters and inference FLOPs, and at what GPU-memory cost?

### FINER Assessment
| Criterion | Score | Justification |
|-----------|-------|---------------|
| Feasible | 4/5 | The e2e JAX codebase already implements chunked TTT, remat-through-time, SWA cache, DCLM loader, per-token eval. Changes are: parameter specs (which weights are inner/outer), a LoRA module, pretrained-init loading, one new inner optimizer. Feasibility risk is GPU memory/latency for "all layers" fast weights and access to DCLM buckets (requester-pays) + GPUs. |
| Interesting | 4/5 | Paper's own ablation (Fig. 4 right) shows larger fast state (6 vs 3 layers) is what makes context scaling match full attention; whether a *low-rank* slow adapter can meta-learn a good TTT initialization from a pretrained model is untested and directly addresses the paper's stated future direction ("initialize from a pre-trained Transformer without TTT"). |
| Novel | 4/5 | No known work meta-learns a LoRA as the outer-loop parameter with full-rank fast weights (to be confirmed in Phase 2). TTT-KVB/LaCT use LoRA/small fast weights (the opposite). |
| Ethical | 5/5 | Public data, no human subjects. |
| Relevant | 4/5 | Cheap meta-training from pretrained checkpoints would remove the paper's main practical limitation (3.4× slower pretraining). |
| **Average** | **4.2/5** | |

### Scope Boundaries
**In scope**: 125M (primary) and 350M/760M (scale check) models; DCLM 8K pretraining-protocol evaluation; optionally Books 32K extension; fast-weight fraction ∈ {last 1/4, last 1/2, all blocks}; inner optimizers ∈ {normalized SGD, Muon-style orthogonalized SGD (no momentum), AdamW ε=1e-8}; LoRA rank ∈ {8, 32, 128} on MLP (optionally attention); memory/latency measurement.
**Out of scope**: NIAH/RULER, decoding evaluations, custom CUDA kernels, models ≥1B, multi-turn persistence of fast weights across sequences (flagged as an open question for the user).
**Key assumptions**: (A1) a pretrained SWA Transformer checkpoint exists or can be trained (paper's `pretrain-*-fa` config at 8K is exactly SWA k=8K); (A2) the JAX e2e codebase is extended rather than re-implemented in PyTorch; (A3) fast weights reset to W0 at every sequence boundary during training and evaluation, as in the paper.

### Sub-questions
1. **Memory/compute**: What checkpointing/recomputation strategy keeps per-GPU memory bounded when the per-chunk carry (fast weights + optimizer state) is 5-50× larger than in the paper, and which strategy is simplest and least error-prone in a JAX scan+remat codebase?
2. **Inner optimizer**: Which normalized inner-loop update (normalized SGD, Muon-style orthogonalization, AdamW) is best when the outer loop differentiates through 8-128 inner steps on large matrices, considering state memory, second-order conditioning, and evidence from TTT/meta-learning literature?
3. **Slow LoRA design**: What rank, scaling convention, placement (MLP vs MLP+attention), and outer learning rate does the LoRA/meta-learning literature support for a LoRA that receives meta-gradients?

### Sub-Question Bindings
1. inherits: models 125M-760M; context 8K-32K; framework JAX e2e; deviations: none
2. inherits: same; inner steps T/b = 8-32 (128 only at 128K, out of primary scope); deviations: none
3. inherits: same; deviations: LoRA placement on attention is an approved widening of the user's "MLP" default, pending user confirmation

---
## Revision after DA Checkpoint 1 (verdict: REVISE; see da_checkpoint1.md)

### Explicit hypothesis and falsifier (DA issue 1)
**H1**: On a pretrained SWA Transformer, most of the meta-learning gain of TTT-E2E over TTT-naive can be captured by a small set of slow parameters (LoRA on attention Q/K/V/O and MLP, all RMSNorm gains, and learned per-tensor inner learning rates), i.e. arm C ≈ arm D (full-slow). **Falsifier**: C ≈ B (TTT-naive) while D ≫ B.
**Corollary of DA issue 1**: LoRA on the fast MLP alone is mathematically a rank-r shift of W0 = W_pre + BA; it is kept as an ablation, not the definition of the arm. Default slow set for arm C = {attention LoRA, MLP LoRA, norm gains, per-tensor inner LR scalars}.

### Scope changes
- **32K extension made mandatory** for arms A, B, C(best), D(best), E (DA issue 5): at 8K, SWA(k=8K) ≡ full attention and the paper reports TTT-E2E ≈ full attention, so 8K-only would be a null result.
- **New arm F** (DA issue 2): paper-style `feed_forward_prime` (last 1/4, cloned from pretrained MLP) as fast, same pretrained init and same meta-training budget as C/D → C-vs-F isolates fast-weight size; D-vs-E isolates the regime/budget confound. Compute reported in H100-hours per arm, including inherited pretraining.
- **Forgetting probe + control** (DA issue 3): after TTT on sequence s, evaluate W_T on a fresh held-out DCLM chunk and report ΔNLL vs W0; sweep decay-toward-W0 λ ∈ {0, 0.05, 0.2} (ΔW ← (1−λ)ΔW per chunk).
- **Memory sub-question** (DA issue 4): at 125M it is answered by arithmetic (≤ 8 GB). The 760M / 32K / all-blocks run (per-device batch 1 via `accum_steps`) is a required arm so the memory question is tested; pre-registered expectation ≈ 20 GB (SGD) / 62 GB (AdamW) fp32 carry per sequence, remedies bf16 ΔW, larger `inner_remat_freq`, `n_state_parallel`. Corrected size ratio vs paper: 1.2–4.9× at 125M, up to ~50× only for 760M-all vs the paper's 125M prime.
- **Matched cost redefined** (DA issue 7): inference FLOPs/token and *deployed (post-merge)* parameter count; LoRA params reported separately as meta-training scaffolding. TTT inference FLOPs for "all blocks fast" are higher than the paper's and are reported.
- **Inner-LR sweep in a common unit** (DA minor): per-tensor RMS update size, so {global-norm, per-tensor-norm, Muon} are comparable; range widened to {1, 3, 10}× the paper-equivalent step. Arm B gets its own inner-LR sweep (dynamic evaluation, Krause et al. 2018, is a strong baseline, not a straw man).
- **Numerics**: keep W0 in fp32 (or cast W0+ΔW in fp32 before the matmul); bf16 is only for ΔW.

### FINER recalibration
Feasible 3/5 (requester-pays data, 24-run second-order sweep, 760M/32K run), Novel 3/5 pending Phase 2 (dynamic evaluation, Meta-SGD/ALFA learned inner LRs, PERK meta-learned LoRA init are prior art), Interesting 4/5 (paper Fig. 4-right is about more fast *layers*, not larger matrices; extrapolation noted), Ethical 5/5, Relevant 4/5 → **3.8/5**.

### Open questions for the user (DA issue 6) — must be answered before implementation
Q1. Reset fast weights at every sequence boundary (paper protocol) **or** persist across turns/sequences (continual-learning protocol; different eval, forgetting across documents)?
Q2. Slow LoRA learned by **meta-gradient through the inner loop** (current reading; "outer loop" of TTT-E2E) **or** by **consolidation** (slow LoRA regressed onto accumulated fast ΔW after each turn)?

### User directive (2026-09-16, after DA checkpoint 1)
Inner-optimizer arms are **both required**: (a) **normalized SGD**, strictly normalized (W ← W − η·g/(‖g‖_F+ε), per tensor and/or global; the paper's clip-to-1 SGD is only a partial normalizer because it leaves steps with ‖g‖<1 unnormalized), and (b) **AdamW with ε=1e-8**, differentiated through, with the literature's known second-order caveats mitigated (mask v=0 at step 1 / warm-start v, keep m,v in the carry, budget 3× copy memory) rather than used as a reason to drop it. Muon-no-momentum remains an optional third arm.
