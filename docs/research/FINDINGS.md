# Big-fast / small-slow TTT-E2E — findings, gaps, and recommendations

Date: 2026-09-16. Produced with the deep-research skill (`lit-review` mode + Phase 1 scoping and two Devil's Advocate checkpoints). Supporting artifacts: `phase1_scoping/` (RQ brief, methodology blueprint, DA checkpoint 1), `phase2_investigation/` (three verified literature layers + source verification), `phase3_analysis/` (synthesis report, DA checkpoint 2). 33 arXiv papers and 23 repos/docs were verified to exist; one ID in the user's source list was wrong (see §6).

## 1. What TTT-E2E actually does (paper + code)

| Design element | As implemented in `test-time-training/e2e` (commit a4fc478) |
|---|---|
| Model | Llama-style Transformer, SwiGLU MLP, RMSNorm pre **and** post, QK-norm, RoPE θ=500K, tied embeddings, Llama-3 tokenizer |
| Sequence mixer | Sliding-window attention, window k=8192. In the TTT (suffix) blocks, `SWA` keeps a k-token KV cache in `eqx.nn.State`; each 1024-token chunk attends to [cache ∥ chunk] with a sliding causal mask. Prefix (frozen) blocks run once over the whole sequence with cuDNN flash local attention. |
| Fast weights | Only `feed_forward_prime`, a *second* SwiGLU MLP added to the last 1/4 of blocks (3/12, 6/24, 8/32). Base MLP widths shrunk (2048→1664 at 125M) so total params match the baseline. Norms, attention, embeddings frozen inside the inner loop. Fast state = 11.5M (125M), 160M (1B), 346M (3B) params ≈ 9–11% of the model. |
| Slow weights | Everything, including the fast-weight init W0 (`spec_outer: ["**"]`). |
| Chunk (mini-batch) size b | 1024 tokens; ablation: b>1K hurts, b<1K unstable/slow. Requires k ≥ b. |
| Inner optimizer | `optax.chain(clip_by_global_norm(1.0), sgd(lr=1.0, momentum=None))` — global norm over **all** fast params jointly, so ‖ΔW_all‖_F = min(‖g‖, 1). This is normalized SGD only when ‖g‖>1. No state. Inner-LR warmup 0.1→1.0 over the first 10% of pretraining; none for extension. |
| Inner/outer loss | Chunk i: loss with W_{i−1}, then one step → W_i. Outer loss = mean over chunks of loss-before-update (Eq. 6). |
| Outer optimizer | AdamW β=(0.9,0.95), wd 0.1, clip 1.0; peak LR 3e-3 (125M) … 8e-4 (3B) for pretraining, 4e-4 for extension; 10% warmup, cosine → 1e-5. |
| Second-order | `eqx.filter_value_and_grad` inside `jax.lax.scan`; outer grad differentiates through the whole scan (true gradients-of-gradients). |
| Memory | Checkpointing **through time**: each chunk step under `jax.remat(nothing_saveable)`, so only the carry (fast weights in fp32 `state_dtype`, inner-opt state, KV-cache state) is saved per chunk boundary; `scan_remat_chunk(inner_remat_freq=n)` adds a second level. Peak ≈ (N/n + n)·(|W|+|v|) + one chunk's activations, per sequence, × vmapped per-device batch (`accum_steps` runs sequences sequentially instead). Fast weights can be sharded across GPUs (`n_state_parallel`). |
| Data / eval | DCLM-Baseline docs ≥8K tokens, 8K context, Chinchilla tokens (2.5B @125M … 54B @3B); extension on Books at 32K–128K with 5% of pretrain tokens and doubled batch. Metric: held-out loss and per-token-index loss curve (`token_nll_loss`). |
| Kernels | Flash/cuDNN attention lacks double backward → used only for the prefix; suffix uses XLA attention. Training is 3.4× slower than full attention at 8K. |

## 2. The proposed design, as understood

Fast = full MLP weight matrices (w1,w2,w3) of the last 1/4 → all blocks, initialized from a **pretrained SWA Transformer** and reset per sequence. Slow = a small meta-learned set: LoRA adapters (+ norm gains + learned inner LRs). Inner loop = normalized SGD / AdamW per 1024-token chunk. Outer loop = AdamW on the meta-gradient through the inner loop. Everything else (SWA k=8K, chunk 1K, DCLM protocol, eval code) reused from e2e.

Interpretation choices I made (confirm or correct): fast weights reset at every sequence boundary; the slow LoRA is trained by the **meta-gradient** through the inner loop (the paper's outer loop), not by consolidation from the fast ΔW.

## 3. Gaps found (Devil's Advocate checkpoint 1, verdict REVISE, no critical issues)

1. **LoRA on the same MLP that is the fast weight is just a rank-r offset of W0** ((W_i + BA)x ≡ W_i x with W0 := W_pre + BA). It cannot learn what the paper's all-slow outer loop learns (attention that decides what to write, norms, the loss landscape). Fix: state the hypothesis H1 "a small slow set (attention LoRA + MLP LoRA + RMSNorm gains + per-tensor inner LRs) captures most of the meta-learning gain", with falsifier C≈B while D≫B; keep MLP-only LoRA as an ablation.
2. **Two variables change at once**: fast/slow inversion *and* pretrained-init + 5% budget vs the paper's from-scratch meta-pretraining. Fix: arm F (paper-style prime MLP as fast, same pretrained init and budget) so C-vs-F isolates fast-weight size; report H100-hours per arm.
3. **No forgetting control or metric** although the pretrained MLP is overwritten (the paper kept it static as "safe storage"). Fix: forgetting probe (ΔNLL of W_T on a fresh DCLM chunk vs W0), decay-toward-W0 sweep λ∈{0,0.05,0.2}, LaCT-style row renorm with the pretrained row norm as fixed point.
4. **Memory question is trivial at 125M** (≤8 GB) and only binds at 760M/32K/all-blocks with AdamW (≈62 GB fp32 carry). Fix: that run is a required arm, pre-registered with the remedies below.
5. **8K-only evaluation is where TTT ≈ full attention** (SWA k=8K ≡ full attention). Fix: 32K Books extension mandatory for arms A, B, C, D, E.
6. **Two intent ambiguities** only you can settle (Q1/Q2 in §7).
7. **"Matched total parameters" is ill-defined** with LoRA added and prime MLP removed. Fix: match on deployed (post-merge) params and inference FLOPs/token; report LoRA params as scaffolding. The merge-after-meta-training property (deployed model = ordinary pretrained Transformer + a per-chunk gradient step) is the strongest practical selling point.

Minor: inner-LR sweep must use a common unit (per-tensor RMS step) so {global-norm, per-tensor-norm, Muon, AdamW} are comparable, and widen upward ({1,3,10}×) or "bigger fast weights store no more" is a self-inflicted null; give arm B (dynamic evaluation, Krause 2018) its own LR sweep; keep W0 in fp32 (a unit-norm step over 57M elements moves each by ~1e-4 ≈ bf16 resolution of a 0.02 weight); add inner-LR warmup for arm C; 3 seeds for headline arms.

## 4. Memory / checkpointing for big fast weights (sub-question 1)

**Cost model (verified in three codebases: e2e, ttt-lm-jax, PyTorch-MAML).** Per sequence, peak ≈ (N/n + n)·(|W| + |v|) + A_step, N = T/b chunks, n = `inner_remat_freq`, |v| = 0 (normalized SGD), |W| (Muon momentum), 2|W| (AdamW). Plus ≥3–4 live |W| buffers for the adjoint and Hessian-vector temporaries.

fp32 carry per sequence, n = round(√N), stateless inner optimizer (× 3 for AdamW):

| Model / fast set | fast params | |W| | 8K (N=8) | 32K (N=32) | 128K (N=128) |
|---|---|---|---|---|---|
| 125M last 1/4 | 14.2M | 0.06 GB | 0.3 | 0.6 | 1.3 |
| 125M all 12 blocks | 56.6M | 0.23 GB | 1.3 | 2.6 | 5.1 |
| 350M all 24 blocks | 207.6M | 0.83 GB | 4.7 | 9.4 | 18.8 |
| 760M last 1/2 | 226.5M | 0.91 GB | 5.1 | 10.3 | 20.5 |
| 760M all 24 blocks | 453.0M | 1.81 GB | 10.3 | 20.5 | 41.0 |
| 1B all 24 blocks | 792.7M | 3.17 GB | 18.0 | 35.9 | 71.8 |
| (paper) 3B prime, 8 blocks | 346M | 1.38 GB | 9.6 | 19.3 | 31.3 |

**Ranked strategies (simplest / least error-prone first), JAX e2e:**
1. Keep the existing scan-through-time remat; set `inner_remat_freq ≈ √N` (3 at 8K, 6 at 32K, 11 at 128K). Zero new code, N → ~2√N copies. Use `accum_steps` so the vmapped per-device batch is 1. Keep fast weights fp32. Prefer a stateless inner optimizer for the biggest arms.
2. Offload the group-boundary W snapshots to pinned host memory: tag the fast-weight carry leaves with `jax.ad_checkpoint.checkpoint_name(..., "fast_w")` and use `save_and_offload_only_these_names([], ["fast_w"], "device", "pinned_host")` on the outer group. Device holds ~n copies, host N/n. Local patch (e2e already dispatches named policies); platform support varies, verify with `Compiled.memory_analysis()`.
3. Shard the fast weights across GPUs with the existing `n_state_parallel` mesh axis before resorting to bf16.
4. Treeverse / nested remat (`eqx.internal.scan(kind="checkpointed")`): O(log N) copies at O(log N)× recompute; only if 2 is unavailable.
5. bf16 fast state: halves copies but risks quantizing away small updates; if used, store ΔW = W − W0 in bf16 with W0 fp32 and form the sum in fp32. Last resort.

Note a paper/code mismatch: the paper says checkpointing through time grows "by a factor of log(T)" (treeverse-style nesting), but the released code does two-level √N grouping. Table values above are computed from the e2e model configs (counts, not estimates); the DA's earlier "≈440M" 760M figure was a back-derivation and is superseded.

**Not recommended and why.** MixFlow-MG (arXiv:2505.00793): shrinks the second-order *activation* term O(|A|), not the T·|W| carry term that dominates at 760M/32K (at 125M nothing binds, so it is unnecessary there too); needs a custom VJP per inner optimizer; no repo. Reversible inner updates (Maclaurin 2015): clipped, momentum-free SGD is not invertible. PyTorch `torch.compile` memory-budget / SAC: compile does not support double backward (pytorch#91469 open since 2022), so it can only cover the frozen prefix. Axolotl offloading: a wrapper over `saved_tensors_hooks`/`save_on_cpu` tied to HF layer structure and first-order fine-tuning. PyTorch-MAML: the correct pattern (checkpoint per inner step) but with a fragile reentrant-checkpoint hack; today use `use_reentrant=False`. RW-TTT: a serving system, first-order, no outer loop.

**Framework decision.** Extend the JAX e2e codebase. A PyTorch port would have to reproduce the scan+remat structure in eager mode with non-reentrant checkpoints, no compile in the meta-gradient region, and math-path attention (fused kernels lack double backward, as GradMem found). Strictly more error-prone for no benefit.

## 5. Inner optimizer (sub-question 2) — both arms required per user directive

**Arm N: normalized SGD (strictly normalized).** Per tensor: W ← W − η · g / (‖g‖_F + 1e-6); optionally the global variant g/(‖g_all‖_F + ε). Unlike the paper's clip-to-1, this normalizes every step. η is expressed as a per-tensor RMS update and swept over {1, 3, 10}× the paper-equivalent step (paper's global cap ≈ 2% relative change per chunk on a 1536×3328 init-scale matrix). Optionally make η a learned per-tensor scalar in the slow set (Meta-SGD / MAML++ LSLR / LaCT `softplus` LR) — stateless, differentiable, and it is the cheapest place to put meta-learnable capacity. Backward of g/‖g‖ is (I − ĝĝᵀ)/‖g‖: well conditioned whenever ‖g‖ ≫ ε. Zero optimizer state, so no extra carry memory. This is what every working differentiate-through TTT system uses (TTT-E2E, TTT-MLP, LaCT's base rule, E²-TTT, Titans/Atlas variants).

**Arm A: AdamW with ε=1e-8, differentiated through.** Known hazards to engineer around rather than avoid: (i) with a cold-start v=g² at step 1, Δ = g/(|g|+ε) and dΔ/dg = ε/(|g|+ε)² — near 0 for |g|≫ε and ~1e8 for |g|≪ε, so the meta-gradient through the first step is per-element dead or exploding; (ii) d√v/dv → ∞ at v=0 (the `higher` library masks this with a backward hook); (iii) with 8–32 steps and β₂=0.999, v̂ stays dominated by a few samples; (iv) m and v (2|W|) ride in the carry, tripling copy memory. Pre-registered mitigations: warm-start m,v from the first chunk's gradient (or a pilot forward), which removes the step-1 singularity; mask v=0 in the backward; carry m,v in fp32; β₂ suited to 8–32 steps (e.g. 0.9); keep ε=1e-8 as requested and log inner- and outer-gradient norms per step with a kill criterion (NaN rate, outer-gradient-norm ratio vs the SGD arm). Treat the 760M/32K AdamW arm as the memory stress test. Existence proofs that differentiating through Adam works at LM scale are in the corpus: MixFlow-MG (2–8 Adam steps, 44M–16B), MASS (2 Adam-LoRA steps, 8B), PERK (4 AdamW steps, 127M–0.5B); none is a TTT system with 8–32 full-rank steps, so this arm is a genuine test, not a known failure. Useful framing (DA checkpoint 2): momentum-free Adam over a few steps is per-element RMS-normalized SGD, so the three arms form a ladder — global norm (paper), per-tensor norm (arm N), per-element norm (arm A).

Caveat for arm N: strict normalization g/(‖g‖_F+ε) has a 1/‖g‖ Jacobian factor when a late, well-fit chunk yields a tiny gradient — the per-tensor analogue of Adam's ε issue, milder because tensor norms are rarely near zero. Pre-register ε (1e-6) and a floor on ‖g‖.

**Optional arm M: Muon without momentum** (LaCT Eq. 9, Moonshot RMS scale 0.2·√max(m,n)). Stateless and fully differentiable, but at b=1024 the five Newton–Schulz iterations cost ≈3× (125M) to 6× (LaCT-size) the matrix's own fwd+bwd, the NS intermediates must be stored or recomputed for the outer backward, and its backward amplifies small-singular-value directions (~500× slope at σ=0 in bf16). LaCT found it helpful at 2K–1M-token chunks; Atlas's ablation found no perplexity gain. Run once at 125M as a hypothesis test.

## 6. Slow LoRA (sub-question 3)

- **Placement**: attention Q/K/V/O LoRA + RMSNorm gains + per-tensor inner LRs as the default slow set; MLP LoRA as an ablation (it is redundant with W0). The SFT results "MLP-only beats attention-only" (Thinking Machines) and "{Wq,Wv} r=8 best" (Hu et al.) are about frozen bases and do not transfer: here the MLP is already trainable in the inner loop.
- **Rank**: sweep r ∈ {16, 64, 256}, start 64. The outer objective is continued-pretraining-scale, where Biderman et al. show full-FT ΔW rank is 10–100× typical LoRA ranks and MLP > attention; PERK used r=256 even on GPT-2-127M.
- **Scaling / init**: rsLoRA α/√r with α fixed (α=16 → scale 2 at r=64); A kaiming-uniform, B=0 so the outer loop starts exactly at the TTT-naive point (clean C-vs-B ablation). Standard α/r collapses gradients at r≥128 (rsLoRA Thm 3.2).
- **Outer LR**: the first-order fine-tuning literature puts LoRA's optimum 4–40× the full-weight optimum (Thinking Machines fit 9.8×; Biderman 4× for CPT, ~10× for IFT; Hu 2e-4 vs 5e-6 ≈ 40×). The relevant full-weight anchor is the paper's 4e-4 extension LR (arm C is an extension-style run), giving a centre near 1e-3–3e-3. The only second-order-LoRA precedent, PERK, meta-learned a rank-256 LoRA init on GPT-2-127M and Qwen2.5-0.5B with outer AdamW **1e-5** (whether tuned is unknown). So the bracket must span both anchors: {3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2} at 125M, {3e-5 … 3e-3} at 760M; keep β=(0.9,0.95), wd 0.1, clip 1.0, 10% warmup, cosine→1e-5. No source measures LoRA LR under second-order gradients, so sweep, do not assume. LoRA is less batch-tolerant than full FT: prefer the paper's pretraining batch (not the doubled extension batch) at the higher LR. LoRA+ (η_B = 4–16 η_A) is an optional 1–2% lever.
- **The sayakpaul gist** (`svd_low_rank_lora.py`) is a post-hoc rank reducer: ΔW = BA → SVD → B'=U√S, A'=√S·Vh, with per-module relative Frobenius error. Use it as an effective-rank diagnostic: train at r=256, truncate to k ∈ {4…64}, evaluate meta-test loss per k without retraining. Fold α/√r into B before truncating (the gist ignores `scaling`); port off the hard-coded `.to("cuda")`; a truncated adapter is not the optimum at that rank.

## 7. Open questions you must answer before implementation

- **Q1** Reset fast weights at every sequence boundary (paper protocol, assumed) **or** persist across turns/sequences (continual learning; different evaluation and a different literature — online meta-learning such as OML/ANML would become the nearest precedents and the forgetting metric would change to backward transfer)?
- **Q2** Slow LoRA learned by the **meta-gradient** through the inner loop (assumed) **or** by **consolidation** from the accumulated fast ΔW each turn?
- **Q3** Which pretrained base: pretrain the paper's 125M `pretrain-125m-fa` config ourselves on DCLM (≈2.5B tokens, a few H100-hours; at 8K it *is* the SWA baseline), request the authors' baseline checkpoint, or port an off-the-shelf HF model?
- **Q4** Hardware and data access: how many GPUs / how much memory, and is a GCP billing project available for the Requester-Pays DCLM/Books buckets?
- **Q5** Fast-weight fraction to start with: last 1/4 (cheapest, keeps the prefix shortcut), 1/2, or all blocks (loses the prefix shortcut; expect ≳3× the paper's training latency)?

## 8. Source list correction
Your list cited arXiv:2505.17895 as "Scalable Meta-Learning via Mixed-Mode Differentiation"; that ID is DataRater (Calian et al.). MixFlow-MG is arXiv:2505.00793 (Kemaev et al., ICML 2025). All other references resolved correctly.

## 9. Pipeline record
Phase 1 RQ brief (FINER 3.8/5 after recalibration) → DA checkpoint 1 (REVISE, 7 major) → Phase 2: three search layers, 56 sources verified, 1 excluded → Phase 3 synthesis (5 themes, 8 cross-paper tensions, 6 gaps) → DA checkpoint 2 (REVISE, 3 major: AdamW over-ruled, outer-LR bracket too narrow, one misattributed number; all three corrected in this document). `lit-review` mode ends here; no APA report was compiled. Implementation is gated on Q1–Q5 in §7.

## 10. Planning round 2 (2026-09-17) — user decisions and answers
Decisions: slow set = LoRA on attention + MLP + norm gains + learned inner LRs (H1); forgetting probe mirrors paper (per-token curve + fresh-chunk ΔNLL); 32K mandatory; ablate fast-weight fraction 1/4 → 1/2 → all; memory plan as §4; both optimizer arms; slow LoRA by meta-gradient (Q2 settled); fast weights reset per sequence (Q1 settled, matches paper); base model from HF if a reputable one exists (Q3); hardware = Princeton della (della-pli H100 80 GB nodes, della-gh GH200 96 GB; SSH needs Duo, so partitions/quota to be confirmed interactively).

Base-model candidates (all full-attention pretrained; at 8K, SWA k=8K is identical, so no architecture change for the meta-training stage):
| Model | Layers / d / ff | All-MLP params (share) | 1/4 blocks | Tokenizer | Note |
|---|---|---|---|---|---|
| meta-llama/Llama-3.2-1B | 16 / 2048 / 8192 | 805M (65%) | 201M | Llama-3 (matches e2e DCLM/Books buckets) | gated, research-OK licence; 128K RoPE |
| HuggingFaceTB/SmolLM2-360M | 32 / 960 / 2560 | 236M (65%) | 59M | own | needs DCLM re-tokenization |
| HuggingFaceTB/SmolLM2-135M | 30 / 576 / 1536 | 80M (59%) | 20M | own | needs re-tokenization |
| Qwen/Qwen2.5-0.5B | 24 / 896 / 4864 | 314M (64%) | 78M | own | needs re-tokenization |
Recommendation: Llama-3.2-1B (tokenizer match removes a whole data-pipeline step). fp32 carry at 32K with n=6 (≈11.3 copies): 1/4 → 9 GB (SGD) / 27 GB (AdamW); all → 36 GB / 109 GB → all-blocks AdamW needs host offload or `n_state_parallel`.

## 11. Implementation status (2026-09-18)

Code: `ttt/` (PyTorch), 140 tests. Results: `docs/results/results.md`. Plan and its
corrections: `docs/superpowers/plans/2026-09-18-pytorch-architecture.md`.

Correctness gates passed:
- Llama-3.2-1B logits reproduced against HuggingFace through our chunked prefix+suffix
  path (correlation 1.000000).
- Inner loop with lr=0 reproduces the plain no-TTT gradient exactly.
- Checkpoint group size is numerically inert; gradients flow through the KV cache across
  chunk boundaries (verified by a detach-and-compare test).

Three things the plan got wrong, each found by running rather than reading:
1. **Inner LR off by 14x.** Derived from the paper's 11.5M-param prime MLP instead of our
   201M fast set. The paper-equivalent per-element step is `1/sqrt(n_fast)`, so the scale
   is not transferable between fast-weight sets of different sizes.
2. **Memory model incomplete.** It counted fast-weight copies only. The binding cost is
   the attention double-backward, which the math SDPA backend forces; backward grows about
   12 GiB per fast block while forward grows under 2 GiB.
3. **RoPE convention.** TTT-E2E uses the interleaved complex form, HuggingFace Llama the
   halves form. Worth 2.0 nats on arm E.

Deviations from the plan, all forced and all recorded:
- 199.6M training tokens instead of 2e9: Della compute nodes have no internet and the
  login-node watchdog kills long streaming jobs.
- Arm C trains at 131,072 tokens per outer step rather than 524,288, because a step costs
  about 127 s at 16 sequences; the planned batch would make a 300-step run take 11 hours
  of pure compute per configuration.
- The fast-weight-fraction ablation stops at 1/4. Half and all-blocks do not fit in 80 GiB.
