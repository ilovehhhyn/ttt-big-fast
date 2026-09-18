## Devil's Advocate Report — Checkpoint 1

**Reviewed**: `research_question_brief.md`, `methodology_blueprint.md` (Phase 1 scoping, 2026-09-16)
**Ground truth used**: brief's baseline-facts table (arXiv:2512.23675v2 + e2e commit a4fc478); user's original chat brief.
**Numbers used below** (125M: d=768, 12 blocks, SwiGLU ffn=2048): MLP params/block 4.72M → last-1/4 14.2M, last-1/2 28.3M, all 56.6M; paper's prime MLP 11.5M. LoRA (A,B on w1,w2,w3): r=8 all-blocks 0.8M; r=128 last-1/4 3.2M; r=128 all-blocks 13.0M. Carry copies per sequence ≈ (T/b)/g + g, g≈√(T/b), fp32.

### Verdict: REVISE

Strengths first, so the criticism is read in proportion: the baseline-facts table is unusually careful and correct on the things that matter (global-norm clip ≡ normalized SGD, remat-through-time, `spec_outer: ["**"]`, k ≥ b); the five-arm design already contains the two controls most reviewers would demand (TTT-naive and big-fast/full-slow); the limitations section admits the 8K ≡ full-attention problem instead of hiding it. The problems below are about whether the study, as scoped, can answer the question it poses.

### Critical Issues (Blocks Progression)
No critical issues identified.

### Major Issues

1. **The LoRA is a rank-r offset of W0; what it can meta-learn is not stated, and its motivation does not exist at the chosen scale**
   - **Type**: Method / Scope
   - **Location**: Brief, "Primary Research Question" and "Topic Area"; Blueprint, arm C row and step 2(iii).
   - **Problem**: With LoRA on the same MLP that is the fast weight, the forward is `(W_i + BA)x` (or `W_i x` with `W_0 := W_pre + BA` merged). Either way the inner loop is *identical* to TTT-E2E whose outer loop is restricted to a rank-r manifold around the pretrained MLP. Everything else the paper's outer loop learns (attention that decides what to write into fast memory, norms, embeddings, the loss landscape in which the inner gradient is taken) is frozen. The paper's own finding is that the all-slow outer loop is what separates TTT-E2E from TTT-naive; the brief offers no hypothesis for why a low-rank init shift of the MLP should capture much of that. Worse, the reason to use a LoRA at all (cheap slow weights / small optimizer state) does not bind at 125M-760M: AdamW state for all 125M params is 1 GB. Arm D (full-slow) is therefore *strictly cheaper to justify* than arm C at every scale in scope, so the headline arm's defining feature is unmotivated inside the study's own boundaries.
   - **Recommendation**: (a) State the hypothesis explicitly: "most of the meta-learning gain on a pretrained base is a low-rank shift of the fast-weight init" — and give it a falsifier (C ≈ D at rank 32 supports it; C ≪ D and C ≈ B falsifies it). (b) Move the LoRA off the MLP where it is redundant with W0, and put the slow capacity where the paper says the outer loop matters: attention Q/K/V/O LoRA (currently only "optional"), all RMSNorm gains (cheap, full-rank), and a *learned per-tensor inner learning rate / per-chunk loss weight* (Meta-SGD/ALFA-style; a few hundred scalars, high leverage, differentiable through the same scan). Make "LoRA on MLP only" one ablation, not the definition of the arm. (c) Reframe the LoRA's purpose as either regularization under a tiny meta-training budget (testable: does D overfit at 125M tokens?) or as the thing that scales to a frozen ≥7B base (then say so, and admit that is out of scope).

2. **Two variables change at once: fast/slow inversion and the pretrained-init + 5%-budget regime**
   - **Type**: Method (confound)
   - **Location**: Brief A1; Blueprint "Sampling" (5% of Chinchilla) and arm E ("released checkpoint, or rerun").
   - **Problem**: Arm E is meta-trained from scratch at 2.5B tokens with second-order gradients; arm C is an extension-style fine-tune of a pretrained model for 125M tokens. Any C-vs-E gap is attributable to budget/regime as much as to fast-weight size. Arm D controls LoRA-vs-full but not regime. There is no arm that runs the paper's fast-weight structure (small prime MLP, static MLP as safe storage) under the *same* cheap regime, so "bigger fast weights help" is not identifiable from "post-hoc TTT-ification of a pretrained model works".
   - **Recommendation**: Add arm F: paper-style `feed_forward_prime` (last 1/4, freshly initialized or cloned from the pretrained MLP) as fast, pretrained init, same 125M-token budget, same slow set as C (LoRA) and as D (full). Then C-vs-F isolates fast-weight size, D-vs-E isolates regime/budget. Also report compute in a single currency (H100-hours, or second-order-FLOPs) per arm, including the pretraining that arm C inherits; otherwise "cheap" (FINER Relevant) is an accounting artifact.

3. **No forgetting control or forgetting metric, although the pretrained MLP is now the fast weight**
   - **Type**: Method
   - **Location**: Blueprint arm C, step 2(vi) (ΔW parametrization is mentioned only as a *memory* trick), Validity table.
   - **Problem**: The paper kept the original MLP static and gave TTT a second MLP precisely so that inner steps cannot erase pretrained knowledge. Arm C overwrites the pretrained MLP with 8-32 unit-norm steps (128 at 128K) and W0 is never restored within a sequence. Because the update norm is fixed while the gradient concentrates on a few rows/tokens, damage can be local and large even when the global relative perturbation looks small. Nothing in the analytical framework would *detect* this: held-out mean loss and per-token curves conflate "learned the context" with "forgot general knowledge".
   - **Recommendation**: (a) Add a forgetting probe: after TTT on sequence s, evaluate W_T on an unrelated held-out DCLM chunk and report ΔNLL vs W0 (paper-style per-token curve can also be split into "late tokens of same doc" vs "fresh doc"). (b) Add at least one inner-loop stabilizer to the sweep: decay toward W0 (`ΔW ← (1−λ)ΔW`, λ∈{0, 0.05, 0.2}) or a smaller inner LR for down-proj w2, or trust-region on ‖ΔW‖. (c) Consider a "safe-storage-preserving" variant that keeps the original MLP static and makes a *copy* of it the fast weight (matched to the paper's structure, at the cost of +MLP params) so the effect of losing safe storage is measured, not assumed away.

4. **Memory sub-question cannot be answered inside the primary scope; the scoping puts the memory question where it is trivial**
   - **Type**: Scope
   - **Location**: Brief Sub-question 1 and "5-50× larger" claim; Blueprint step 2(vi), step 4; Limitations line 1.
   - **Problem**: Using the brief's own copy formula, fp32 carry for *all* 12 blocks at 125M/8K is ≈6 copies × 0.23 GB ≈ 1.3 GB (SGD) or 3.9 GB (AdamW). At 125M/32K: 2.6 / 7.7 GB. The "problem" the user flagged appears only at 760M/32K (≈20 GB SGD, ≈62 GB AdamW per vmap'd sequence) or ≥1B/128K, which the scope makes "optional" and "one confirmation run". Also, at 125M the fast-weight size ratio to the paper is 1.2-4.9×, not "5-50×"; 50× only arises comparing 760M-all to the paper's 125M prime. As written, sub-question 1 will be answered by arithmetic ("it fits") and produce nothing the user asked for.
   - **Recommendation**: Either (a) demote sub-question 1 to a planning table and say the primary study does not stress memory, or (b) make the 760M/32K all-blocks run (with AdamW inner, per-device batch 1 via `accum_steps`) a *required* arm so the memory question is actually tested, and pre-register the expected 80 GB overrun and the remedy (bf16 ΔW, higher `inner_remat_freq`, `n_state_parallel`). Correct the "5-50×" figure per scale.

5. **Primary evaluation sits in the regime where TTT's advantage over SWA is ~0**
   - **Type**: Method / Scope
   - **Location**: Brief scope ("DCLM 8K ... optionally Books 32K"); Blueprint Limitations line 2 and step 5 ("Optional").
   - **Problem**: At 8K with k=8K, SWA ≡ full attention and the pretrained model already has the whole context. The paper's separation between SWA and TTT-E2E appears at ≥16K; at 8K TTT-E2E ≈ full attention. So the primary experiment measures dynamic-evaluation gain on top of a model that lacks nothing, and the most likely outcome is "all arms within 0.01 nats" — uninformative about the RQ's stated interest (per-token loss *decay* and context scaling). The blueprint knows this and still makes 32K optional.
   - **Recommendation**: Make the 32K extension (paper's Books protocol, or long-doc DCLM) mandatory for arms A, B, C(best), D(best), E. It doubles as the only regime where the memory question (issue 4) starts to bind.

6. **The user's intent is not resolved on two points that change the problem**
   - **Type**: Scope
   - **Location**: Brief A3 and "Out of scope: multi-turn persistence ... flagged as an open question for the user".
   - **Problem**: (a) "gets the context and new knowledge every turn" plausibly means fast weights persist across turns/sequences — that is continual learning with a different evaluation (no per-sequence reset, forgetting across documents), not the paper's per-sequence reset. (b) "slow weights are a LoRA layer that is updated via gradient descent *from the large fast weights*" can be read as a fast→slow consolidation/distillation rule (slow LoRA regressed onto the accumulated fast ΔW), not a MAML meta-gradient through the inner loop. The brief silently chose the meta-gradient reading (well supported by "outer loop / inner loop / refer to the paper", but not certain). Flagging is not resolving; Phase 2 literature search and Phase 3 design differ materially under each reading.
   - **Recommendation**: Ask the user two yes/no questions before Phase 2: (1) reset fast weights at every sequence boundary (paper) vs persist across turns; (2) slow LoRA learned by meta-gradient through the inner loop (current design) vs by consolidation from fast ΔW. Record the answers in A3/A4.

7. **"Matched total parameters" is not well-defined as written**
   - **Type**: Method (construct validity)
   - **Location**: Brief RQ ("at matched total parameters and inference FLOPs"); Blueprint Validity "Matched cost: arm E and C at equal total params".
   - **Problem**: Arm C = full 125M pretrained model + LoRA (up to 13.0M extra at r=128 all-blocks, i.e. +10%); arm E = 125M with a shrunk base MLP (1664) plus prime MLP; arm A = 125M. These cannot all be "equal total params". The LoRA can, however, be merged into W0 after meta-training, so the *deployed* arm C has exactly arm A's parameter count and FLOPs — which is a genuine strength the brief does not claim.
   - **Recommendation**: Define matching as: inference FLOPs/token and deployed (post-merge) parameter count; report LoRA parameters separately as meta-training scaffolding; note that arm E cannot be matched to A/C simultaneously and pick which pair is matched. State that TTT inference cost for "all blocks fast" is higher than the paper's (every block's MLP takes a gradient step per chunk) and include it in the FLOPs/token column.

### Minor Issues
- **Inner-LR sweep and normalizer scope are confounded** (Blueprint step 3). Per-tensor norm-1 over 36 tensors is ≈6× the global norm-1 step; Muon's RMS-matched scale is a third unit. Express the inner LR in a common unit (per-tensor RMS update, as Muon does) so {global, per-tensor, muon} are comparable, and widen the range upward ({1, 3, 10} in that unit): with a fixed global step norm spread over 5× more parameters, the sweep must be able to find larger effective steps or "bigger fast weights store no more" becomes a self-inflicted result. Ground truth: the paper's clip is a global norm over *all* fast params jointly (brief, "Inner optimizer" row).
- **Arm B (TTT-naive) is a straw man unless its inner LR is tuned separately** (Blueprint arm B). Dynamic evaluation of full weights on a pretrained LM (Krause et al. 2018) is a strong known baseline; give B its own LR/normalizer sweep, and cite it as prior art (it also lowers the "Novel" score for arm B itself, see below).
- **bf16 W0 + bf16 ΔW forward** (step 2(vi)): a unit-norm global step over 56.6M params moves each element by ~1e-4 on average, comparable to the bf16 resolution of a typical |W0|≈0.02 element (~8e-5). Keep W0 in fp32 or form W0+ΔW in fp32 inside the matmul input cast, or small updates are quantized away.
- **Inner-LR warmup** (brief "Inner optimizer" row): the paper warms 0.1→1.0 over 10% of *pretraining*, none for extension. Arm C starts an un-meta-trained pretrained model at lr=1; arm B will show whether that diverges. Add inner-LR warmup to arm C's config as a knob rather than assuming extension defaults.
- **AdamW inner with ε=1e-8, differentiated through** (brief sub-question 2): the second-order gradient through `g/(√v+ε)` is ill-conditioned at the first step (v≈g²) and doubles carry memory. Pre-commit to a single-step-bias-corrected variant or to ε≈1e-3, and note that with no momentum across 8 steps Adam degenerates toward sign-SGD.
- **Seeds** (Validity "2 seeds for the headline arm"): with a 125M-token meta budget, seed variance is likely above the paper's 0.001-nat noise floor; budget 3 seeds for C and D at the chosen rank, and report the spread.
- **Muon without momentum on a single 1024-token chunk gradient** orthogonalizes a low-rank noisy gradient to a full-spectrum update; flag this as a hypothesis to test (may help or hurt), not a default.
- **Data access** (Blueprint "Sources"): both buckets are Requester-Pays and the 125M-token protocol needs the `Evaluator` split unchanged; add a one-line cost/permission check to Feasible before committing the sweep.

### Observations
- The merge-after-meta-training property (issue 7) is the strongest *practical* selling point of arm C: the deployed model is an ordinary pretrained Transformer plus a per-chunk gradient step. Lead with it.
- If issue 1(b) is adopted, the study becomes "which few slow parameters make dynamic evaluation of a pretrained Transformer meta-learnable?" — a sharper, more defensible question than fast/slow size inversion per se.
- The paper's 3.4× training slowdown is the *reference* per-token cost; arm C with all blocks fast (no prefix shortcut, XLA attention everywhere, second-order through 12 blocks) is plausibly 3-4× worse per token than that. The blueprint acknowledges it (Limitations line 3) but FINER Feasible 4/5 and Relevant 4/5 do not. At 5% tokens the *total* is still below one pretraining run; say so with a number (estimated H100-hours per arm; 24 sweep runs ≈ 3B meta-training tokens).
- Sub-question 3's "approved widening" to attention LoRA is listed as "pending user confirmation" — issue 1 argues it should be the default.

### Strongest Counter-Argument
"At 125M/8K, this study is dynamic evaluation of a pretrained LM with a low-rank tweak to the initialization, in a regime where sliding-window attention already sees the whole context. The paper's result that all-slow meta-learning is what separates TTT-E2E from TTT-naive is exactly the capacity the LoRA removes, and the memory/latency issue the user raised only appears at scales and contexts the study makes optional. Whatever it finds, it will not say whether big fast weights with small slow weights work where they would matter."

### What's Missing
- A stated mechanism/hypothesis for what a rank-r MLP offset can meta-learn (issue 1) and a falsifier for it.
- A regime-matched small-fast control (arm F, issue 2) and a single compute currency across arms.
- A forgetting metric and a forgetting control (issue 3).
- A required run in the regime where either memory or context-scaling actually binds (issues 4, 5).
- Resolution of the two intent ambiguities (issue 6).
- Prior art that lowers "Novel 4/5" *before* it is scored: dynamic evaluation (Krause et al. 2018) = arm B; meta-learned LoRA initializations (MAML/Reptile with adapters), learned inner learning rates (Meta-SGD, ALFA), and TTT-E2E's own stated future direction ("initialize from a pre-trained Transformer without TTT"), which the brief cites as support but which is closer to arm D than to arm C. Scoring novelty at 4/5 "to be confirmed in Phase 2" inverts the order; hold at 3/5 until the lit review returns.
- FINER recalibration: Feasible 3/5 (requester-pays data, 24-run second-order sweep, 760M/32K run unpriced), Novel 3/5 (pending), Interesting 4/5 is acceptable but the cited Fig. 4-right evidence is about *more fast layers of a prime MLP*, not larger matrices; note the extrapolation.

### Stress Test Results
| Test | Result |
|------|--------|
| Remove strongest source — does argument hold? (drop the paper's "larger fast state ⇒ better scaling" ablation) | No — the case for *bigger* fast weights then rests only on intuition; arm F is needed to supply the evidence internally. |
| Flip the research question — is opposing view credible? ("small fast, all slow is the right allocation; LoRA-slow will match TTT-naive") | Yes — it is the paper's own position and is fully consistent with issue 1; the design can detect it only if arms B, D and F are all run and tuned. |
| Apply to different context — does finding generalize? (≥1B, 128K, persistent fast weights) | No — precisely the contexts where memory and forgetting bind are out of scope; a 125M/8K result does not transfer (brief admits regime change ~760M). |
| "So what?" — is the significance justified? | Partially — the merge-after-meta-training property and cheap post-hoc TTT-ification of pretrained models are significant *if* the 32K run and forgetting probe are made mandatory; at 8K-only the result is likely a null within noise. |
