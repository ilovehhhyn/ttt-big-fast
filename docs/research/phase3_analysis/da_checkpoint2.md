## Devil's Advocate Report — Checkpoint 2

**Reviewed**: `synthesis_report.md` against `search_layer_lora.md`, `search_layer_inner_optimizer.md`, `search_layer_memory_checkpointing.md`, `source_verification_report.md`, the revised `research_question_brief.md` (including the post-CP1 user directive), and my own `da_checkpoint1.md`.
**Arithmetic re-done here**: every memory-table cell, both Muon overhead figures, the NS5 FLOP count, the LR brackets, and the LaCT chunk count. Numbers that reproduce are not listed; only discrepancies and unsourced numbers are.

### Verdict: REVISE

Strengths, so the criticism reads in proportion: the memory table reproduces from the copy formula to within rounding; the contradiction table and the tension inventory are honest about what was and was not scanned; the synthesis says plainly that no audit was run, that the 350M row is interpolated, and that the corpus is conditional on the meta-gradient/reset reading. CP1 issues 1, 3 and 4 were addressed with new evidence (Block et al., LaCT renorm and Titans decay as stabilizers, the planning table), not restated. The problems below are concentrated in T2 and one paragraph of T3.

### Critical Issues (Blocks Progression)
No critical issues identified. Major issue 1 must nonetheless be fixed before Phase 4, because the report compiler would otherwise carry a recommendation that contradicts the pre-registered design.

### Major Issues

1. **AdamW is "ruled out" on evidence that supports only a warning, in direct conflict with the user directive**
   - **Type**: Evidence / Bias (survivorship, cherry-picking, misapplied source)
   - **Location**: Manifest C-005; T2 ¶1 and Recommendation ("Do not run AdamW inner except as a single ε=1e-3 sanity point"); Convergence Map "Strong: T2 stateless SGD-family".
   - **Problem**: The brief's user directive (2026-09-16, after CP1) makes AdamW ε=1e-8 a required arm *with* mitigations (mask v=0 or warm-start v, keep m,v in the carry, budget 3× memory). The synthesis read that brief and still recommends not running the arm. Its four pieces of evidence do not carry that weight:
     (a) "Every differentiate-through TTT system uses SGD-family" is convergence of *design choices* from at most three research lineages (Sun→Tandon; Behrouz; Zhang), one of which states explicitly that Adam was "left to future work" (search_layer_inner_optimizer §5). Nobody tested it; that is absence, not a negative result.
     (b) The Metz et al. quote ("strongly biased or exploding norm") is about unroll length and applies to *every* arm, including the SGD default. Using it as Adam-specific evidence is a misapplication.
     (c) The agent's derivative dΔ/dg = ε/(|g|+ε)² assumes a cold-start v = g² at step 1. The directive's mitigation (warm-start v, or `higher`'s mask) removes exactly that singularity: with v warm-started the step-1 Jacobian is ≈1/(√v+ε), which is benign. The synthesis argues against the unmitigated variant the user did not ask for.
     (d) In-corpus counter-evidence is omitted: MixFlow-MG differentiates through 2–8 Adam steps at 44M–16B parameters (search_layer_memory §2), MASS differentiates through 2 Adam-LoRA steps on 8B using it, and PERK through 4 AdamW steps at 127M–0.5B. The synthesis cites MixFlow-MG only for memory and calls PERK "the one dissent". Three systems in the corpus doing the thing the synthesis calls impractical is not one dissent.
     The synthesis is also internally inconsistent: Gap 3 says no paper compares SGD/normalized SGD/Muon/Adam inside an unrolled loop and calls the sweep "itself a contribution", then T2 recommends deleting the Adam arm from that sweep.
   - **Recommendation**: Rewrite C-005 and the T2 recommendation as: "AdamW ε=1e-8 is a required arm with pre-registered mitigations (warm-start m,v from the first chunk's gradient or a pilot forward; mask v=0 in the backward as `higher` does; m,v carried in fp32; 3× carry budget from the planning table) and a pre-registered kill criterion (NaN rate, inner-gradient norm blow-up, outer-gradient norm ratio vs the SGD arm reported per step)." Add the observation that momentum-free Adam over 8–32 steps is per-element RMS-normalized SGD, so the arm is the per-element counterpart of the per-tensor default and is scientifically informative whichever way it goes. Cite MixFlow-MG, MASS and PERK as existence proofs of differentiate-through-Adam at LM scale. Downgrade the convergence-map entry to "Moderate: no TTT system has tested Adam".

2. **The outer-LR bracket drops its own low anchor, and the resolution mis-describes PERK**
   - **Type**: Evidence (contradiction explained away) / internal inconsistency
   - **Location**: Manifest C-008 ("PERK/FocuSFT's 1e-5 as the low anchor"); T3 ¶Outer LR; Contradictions table row 2; CP-002.
   - **Problem**: The resolution states PERK and FocuSFT are "0.5B–7B models fine-tuned from instruction-tuned checkpoints", so 1e-5 is a full-FT-scale LR. The LoRA search layer (§6) says PERK trained GPT-2-127M as well as Qwen2.5-0.5B, meta-learning a rank-256 LoRA init with outer AdamW 1e-5. That is the closest precedent to arm C in both structure (meta-gradient LoRA) and scale (127M), and it sits two orders of magnitude below the synthesis's centre of 3e-3. The recommended 125M bracket {3e-4, 1e-3, 3e-3, 1e-2} excludes 1e-5 and 1e-4 entirely, contradicting C-008, which promised to use 1e-5 as the low anchor. Whether PERK tuned its LR is unknown; the honest resolution is "unknown", not "different regime".
   - **Recommendation**: Correct the PERK description. Extend the 125M bracket downward to include 1e-4 (and 3e-5 if budget allows) so the meta-gradient precedent is inside the sweep; state that the 10× rule is first-order fine-tuning evidence and that the only second-order LoRA precedent chose 1e-5. Make C-008 and the T3 bracket agree.

3. **Provenance of the 760M memory row is misattributed, and two derived numbers have no source**
   - **Type**: Evidence (citation)
   - **Location**: T1 planning table and the sentence "The 760M/32K figure (≈20 GB SGD / 62 GB AdamW) is pre-registered in the brief <ref:arxiv:2512.23675>".
   - **Problem**: The paper does not contain 20 GB or 62 GB; those are my CP1 arithmetic, copied into the brief. Citing them to arXiv:2512.23675 will produce a false citation in the Phase 4 report. Further: 62 GB ≠ 3 × 20 GB (the synthesis's own AdamW rule gives 60 GB); "≈1.77 GB per copy", "≈440M fast params" and "≈58% of the model" are back-derived from the 20 GB figure and appear in no search layer. The memory search layer does contain a paper-sourced 760M number ("hidden state 88M vs 18M at 760M") that was not used. The 350M row is correctly labelled as an estimate.
   - **Recommendation**: Cite the 760M row to `da_checkpoint1.md`/brief, not the paper; reconcile 60 vs 62; either read the e2e 760M config (d_model, intermediate size, block count) and replace the back-derived 440M with a real count, or label all three 760M numbers "DA estimate, unverified". Say so in the table caption.

### Minor Issues
- **10× rule compresses a 4–40× range.** Hu et al. is 2e-4 vs ~5e-6 (≈40×, per search_layer_lora §1); Biderman's CPT setting (the regime the synthesis itself says applies) is 4e-5 vs 1e-5 (4×). Report the range; it argues for a wider bracket, which also fixes Major 2.
- **"Three code readings converge" overstates independence.** e2e and ttt-lm-jax share an author lineage and the same two-level scan; PyTorch-MAML implements n=1 only and does not exhibit the √N formula. Say "two related JAX implementations plus one single-level PyTorch analogue".
- **log(T) vs 2√N is presented as consistent; it is a discrepancy.** The paper's remark implies treeverse-style nesting; the released code does √N grouping. Note it as a paper/code mismatch, not as convergence.
- **Strict normalization reintroduces the conditioning problem the paper's clip avoids.** The synthesis calls the backward of g/‖g‖ "well conditioned when ‖g‖≫ε"; the directive's strictly-normalized arm has a 1/‖g‖ Jacobian factor for small chunk gradients (late, well-fit chunks), which is the per-tensor analogue of the Adam ε singularity the synthesis uses to disqualify Adam. State the asymmetry: per-tensor norms are rarely near zero, per-element ones often are, so the degree differs, but the structure is the same. Add ε and a floor to the strict-normalized arm's pre-registration.
- **Muon cost is expressed relative to the wrong denominator.** "≈3× / ≈6× the matrix's own fwd+bwd" is correct arithmetic, but the relevant denominator is the full training step (prefix blocks, attention, second-order backward through the scan). Relative to that, Muon's overhead at 125M is plausibly 1.5–2×, not "decisive". Keep the ablation; soften "decisive".
- **MixFlow-MG dismissal is right at 760M for the wrong reason at 125M.** With carry = 1.3 GB at 125M/8K, the second-order activation term through 12 fast blocks likely dominates, which is MixFlow-MG's regime. Nothing binds at 125M so the conclusion stands, but the sentence "our binding term is T·|W|" should be scoped to 760M/32K. Also update "|v|=0" now that the AdamW arm is required.
- **Unverified venue graded A.** PERK is "ICLR 2026 per agent" in the verification report; the matrix grades it A. Mark as "A (unverified venue)" or B.
- **Uncounted counts.** "≈27 load-bearing" and "six load-bearing 2026 preprints" have no enumeration. List them or drop the numbers.
- **Attribution merge in T4.** "W_i = W0 + ΣΔ_j" is In-Place TTT; "rank-one accumulated writes" is TTT-NTP (search_layer_memory §7). Separate them.
- **Unsourced mechanism claim.** "the attention LoRA's job is deciding what gets written into fast memory" has no source; label as the study's hypothesis (it is H1's mechanism), not a finding.

### Observations
- **CP1 tracking.** Issues 1, 3, 4 were addressed with new evidence. Issues 2, 5, 7 are design matters outside a lit-review and were correctly left to the brief. Issue 6 (reset vs persist, meta-gradient vs consolidation) was restated in Synthesis Limitations, not resolved; the entire corpus and T3/T4/T5 are conditional on the answer the user has not given. The Phase 4 report should carry this as a front-matter gate, as the synthesis recommends.
- **Momentum-free Adam is per-element normalized SGD.** Over 8–32 steps with bias correction, Δ_t ≈ g_t / RMS(g_1..t) elementwise. The three required arms then form a clean ladder: global norm (paper), per-tensor norm (directive a), per-element norm (directive b). Framing it this way makes the AdamW arm a hypothesis test rather than a risk to be avoided.
- **The Muon contradiction is genuinely resolved** (conditional on chunk-gradient rank); the LoRA-placement and rank contradictions are genuinely resolved (frozen-base SFT vs trainable-MLP inner loop; SFT vs CPT scale). The outer-LR contradiction is the one that was explained away (Major 2). The safe-storage tension is correctly left open.
- **Survivorship question (caller's).** Yes: the corpus is TTT papers that exist, and those are SGD-family because TTT-Linear deferred Adam and its descendants inherited the choice; Titans/Atlas chose GD for their own reasons. The synthesis's "Strong" rating turns that lineage into evidence. Under an inclusion rule that admitted learned-optimizer / meta-learning systems that differentiate through adaptive updates as a matter of course (MixFlow-MG, DataRater, PERK, L2O), T2 would read "adaptive inner updates are routine with known mitigations; untested in TTT".

### Strongest Counter-Argument
"The synthesis converts the absence of Adam from five TTT papers of three research lineages into a prohibition, while its own corpus contains three systems that differentiate through Adam at LM scale, and it removes the only second-order-LoRA precedent's learning rate from the bracket by misdescribing that precedent's model size. Where the evidence is thinnest, it is stated most confidently."

### What's Missing
(Candidates from my own knowledge, not in the verified corpus; each needs a Phase 2 addendum check before use.)
- **L2O mitigations, not just L2O pathologies.** Metz et al. and Andrychowicz et al. are cited only for the warning quotes; their fixes (gradient preprocessing, variational/ES-smoothed meta-gradients, truncation schedules) and later learned optimizers (e.g. VeLO) are the standard toolkit for differentiating through adaptive updates and belong in T2.
- **Online meta-learning for continual learning** (OML, Javed & White 2019; ANML, Beaulieu et al. 2020): meta-learn a small slow representation so that many inner SGD steps on a large fast set do not forget. If the user picks persistence (Q1), these are the nearest precedents for both T4 and T5, and the novelty score would need re-scoring.
- **Dynamic evaluation** (Krause et al. 2018, 2019) is still unverified; arm B is defined by it.
- **Big-fast precedents with pretrained full weights**: TTT on nearest neighbours (Hardt & Sun 2023) fine-tunes all weights of a pretrained GPT at test time and reports gains and forgetting, first-order; fast-weight programmers (Schmidhuber 1992; Irie et al. 2021) for T5.
- **DataRater (arXiv:2505.17895)** was excluded as a wrong ID, but it is itself evidence of differentiating through Adam inner steps at scale; re-admit it under its correct role.
- **e2e 760M configuration** (d_model, ffn, blocks) so the 760M memory row is counted, not back-derived.
- **Continual-learning forgetting metrics** (backward transfer, average forgetting) if persistence is chosen; the current probe (ΔNLL on a fresh chunk) is a per-sequence metric only.

### Stress Test Results
| Test | Result |
|------|--------|
| Remove strongest source — does argument hold? (drop TTT-E2E code) | T1: Yes, the formula survives via ttt-lm-jax (same lineage, so weaker than claimed). T2: the SGD "convergence" drops to four systems in two lineages and the prohibition on Adam collapses; the warning survives. |
| Flip the research question — is opposing view credible? ("Adam with warm-started state is a fine inner optimizer under meta-gradients") | Yes — MixFlow-MG, MASS and PERK are in-corpus existence proofs; nothing in the corpus tests the opposite. |
| Apply to different context — does finding generalize? (persistence / consolidation readings of the user's intent) | No — the synthesis admits T3/T4/T5 need a different literature; T2 and T1 largely transfer. |
| "So what?" — is the significance justified? | Partially — T1 (memory is arithmetic at 125M, binds only at 760M/32K) and T5 (no exact precedent) are well supported; T2's practical recommendation is not, and it is the one that shapes the required arms. |
