## Synthesis Report — Big-fast / LoRA-slow TTT-E2E (Phase 3)

**Mode**: deep-research `lit-review`, Phase 3 (Analysis). **Date**: 2026-09-16. **Corpus**: Phase 1 brief (post-DA revision), methodology blueprint, DA checkpoint 1, three Phase 2 search layers, source verification report (42 verified; arXiv:2505.17895 excluded, MixFlow-MG cited as arXiv:2505.00793). All sources are Level VI (single empirical ML studies / code / vendor docs); evidence weight below is therefore expressed as *convergence count* and *grade* (A peer-reviewed, B preprint/blog) rather than hierarchy level. Numerical claims are as reported by the Phase 2 agents against fetched text/code; the verification report states they were not independently re-read by the orchestrator, so every number below is "per agent report, cited to source". No audit step was run by this agent.

### Claim Intent Manifest (v3.8, emitted before prose)

```json
{
  "manifest_version": "1.0",
  "manifest_id": "M-2026-09-16T00:00:00Z-p3sy",
  "emitted_by": "synthesis_agent",
  "emitted_at": "2026-09-16T00:00:00Z",
  "claims": [
    {"claim_id": "C-001", "claim_text": "Per-sequence peak carry memory under scan+remat through time is approximately (N/n + n)(|W|+|v|) plus one step's activations, minimized near n = sqrt(N).", "intended_evidence_kind": "empirical", "planned_refs": ["web:e2e-repo", "web:ttt-lm-jax", "arxiv:2407.04620"]},
    {"claim_id": "C-002", "claim_text": "At 125M the memory question is trivial (<8 GB); it binds only at 760M/32K all-blocks, especially with AdamW inner state.", "intended_evidence_kind": "empirical", "planned_refs": ["arxiv:2512.23675"]},
    {"claim_id": "C-003", "claim_text": "MixFlow-MG attacks the second-order activation term, not the T*(|W|+|v|) carry term, so it gives little benefit when N is large and W is large.", "intended_evidence_kind": "theoretical", "planned_refs": ["arxiv:2505.00793"]},
    {"claim_id": "C-004", "claim_text": "torch.compile cannot be used in the meta-gradient region because double backward is unsupported; JAX scan+remat is the simplest, least error-prone path.", "intended_evidence_kind": "empirical", "planned_refs": ["web:pytorch-issue-91469", "web:pytorch-ac-blog", "web:pytorch-maml"]},
    {"claim_id": "C-005", "claim_text": "All practical differentiate-through TTT systems use stateless SGD-family inner updates; AdamW with eps=1e-8 is ill-conditioned at the first inner step and triples carry memory.", "intended_evidence_kind": "empirical", "planned_refs": ["arxiv:2512.23675", "arxiv:2407.04620", "arxiv:2505.23884", "web:higher", "arxiv:1810.10180"]},
    {"claim_id": "C-006", "claim_text": "Muon's benefit inside TTT is contested: LaCT reports a gain, Atlas's ablation shows none; Muon's cost at chunk 1024 is several times the per-matrix forward-backward and its double backward is the most ill-conditioned candidate.", "intended_evidence_kind": "empirical", "planned_refs": ["arxiv:2505.23884", "arxiv:2505.23735", "web:muon-blog", "arxiv:2502.16982"]},
    {"claim_id": "C-007", "claim_text": "LoRA on the fast MLP alone is a rank-r shift of W0; the slow set should include attention LoRA, norm gains and learned per-tensor inner LRs.", "intended_evidence_kind": "theoretical", "planned_refs": ["arxiv:1707.09835", "arxiv:1810.09502", "arxiv:2505.23884"]},
    {"claim_id": "C-008", "claim_text": "For a CPT-scale outer objective, rank should be swept up to 256 with rsLoRA scaling, and the outer LR bracketed by the 10x rule against TTT-E2E's full-weight LR, with PERK/FocuSFT's 1e-5 as the low anchor.", "intended_evidence_kind": "empirical", "planned_refs": ["arxiv:2312.03732", "arxiv:2405.09673", "web:thinking-machines-lora", "arxiv:2507.06415", "arxiv:2605.09932", "arxiv:2512.23675"]},
    {"claim_id": "C-009", "claim_text": "No source reports a forgetting probe for large pretrained fast weights; LaCT's row renorm and Titans/Atlas decay are the only stabilizers in the corpus and both are meta-learned or fixed, not evaluated for knowledge retention.", "intended_evidence_kind": "empirical", "planned_refs": ["arxiv:2512.23675", "arxiv:2505.23884", "arxiv:2501.00663", "arxiv:2505.23735"]},
    {"claim_id": "C-010", "claim_text": "No paper in the corpus meta-learns a LoRA-only slow set with full-rank pretrained MLPs as fast weights; the closest are PERK (LoRA slow, LoRA fast), LaCT (40% fast, all slow) and In-Place TTT / TTT-NTP (large fast, first-order).", "intended_evidence_kind": "empirical", "planned_refs": ["arxiv:2507.06415", "arxiv:2505.23884", "arxiv:2604.06169", "arxiv:2606.21803", "arxiv:2605.09932"]}
  ],
  "manifest_negative_constraints": [
    {"constraint_id": "MNC-1", "rule": "No claim that a strategy 'works' at 760M/32K before it is measured; all memory numbers are planning estimates."},
    {"constraint_id": "MNC-2", "rule": "No citation of Krause et al. 2018 (dynamic evaluation) with a ref slug; it is not in the verified corpus."}
  ]
}
```

### Literature Matrix

Themes: T1 memory/checkpointing, T2 inner optimizer, T3 slow LoRA, T4 forgetting/stability, T5 precedent/novelty. S = supports the theme's recommendation, C = contradicts or complicates, I = informs (data only), -- = silent.

| Source | T1 | T2 | T3 | T4 | T5 | Method | Grade |
|---|---|---|---|---|---|---|---|
| TTT-E2E, Tandon et al. (2025) <!--ref:arxiv:2512.23675--><!--anchor:section:Appendix%20B--> | S | S | I | C | I | code + LM ablations | B (ground truth) |
| TTT-Linear/MLP, Sun et al. (2024) <!--ref:arxiv:2407.04620--><!--anchor:section:Appendix%20C--> | S | S | -- | -- | I | code + LM | B |
| LaCT, Zhang et al. (2025) <!--ref:arxiv:2505.23884--><!--anchor:section:Eq.8-9--> | I | C | -- | S | S | code + LM/NVS | B |
| Muon (Jordan blog) <!--ref:web:muon-blog--><!--anchor:none:--> / Moonshot (2025) <!--ref:arxiv:2502.16982--><!--anchor:section:Lemma%201--> | -- | I | -- | -- | -- | blog / LM pretrain | B |
| Titans (2024) <!--ref:arxiv:2501.00663--><!--anchor:none:--> / Atlas (2025) <!--ref:arxiv:2505.23735--><!--anchor:section:Table%206--> | -- | C | -- | S | I | LM | B |
| MixFlow-MG, Kemaev et al. (2025) <!--ref:arxiv:2505.00793--><!--anchor:none:--> | C | -- | -- | -- | -- | JAX meta-learning | A (ICML) |
| PyTorch AC blog <!--ref:web:pytorch-ac-blog--><!--anchor:none:--> / issue #91469 <!--ref:web:pytorch-issue-91469--><!--anchor:none:--> | S | -- | -- | -- | -- | vendor docs / issue | B |
| Axolotl offloading docs <!--ref:web:axolotl-docs--><!--anchor:none:--> | I | -- | -- | -- | -- | vendor docs | B |
| PyTorch-MAML <!--ref:web:pytorch-maml--><!--anchor:section:models/maml.py%20L158-185--> | S | -- | -- | -- | -- | code | B |
| JAX memory-spaces docs <!--ref:web:jax-memory-spaces--><!--anchor:none:--> | S | -- | -- | -- | -- | vendor docs | B |
| Maclaurin et al. (2015) <!--ref:arxiv:1502.03492--><!--anchor:none:--> | C | -- | -- | -- | -- | theory + code | A (ICML) |
| higher (facebookresearch) <!--ref:web:higher--><!--anchor:section:higher/optim.py--> | -- | S | -- | -- | -- | code | B |
| Metz et al. (2018) <!--ref:arxiv:1810.10180--><!--anchor:none:--> | -- | S | -- | -- | -- | analysis | A (ICML) |
| MAML++ (2018) <!--ref:arxiv:1810.09502--><!--anchor:none:--> / Meta-SGD (2017) <!--ref:arxiv:1707.09835--><!--anchor:none:--> | -- | S | S | -- | I | few-shot | A / B |
| E²-TTT (2026) <!--ref:arxiv:2608.21308--><!--anchor:none:--> | -- | S | -- | I | -- | LM | B |
| LoRA, Hu et al. (2021) <!--ref:arxiv:2106.09685--><!--anchor:section:4.1--> | -- | -- | I | -- | -- | LM FT | A (ICLR) |
| rsLoRA (2023) <!--ref:arxiv:2312.03732--><!--anchor:section:Thm%203.2--> | -- | -- | S | -- | -- | LM FT | B |
| LoRA+ (2024) <!--ref:arxiv:2402.12354--><!--anchor:none:--> / DoRA (2024) <!--ref:arxiv:2402.09353--><!--anchor:none:--> | -- | -- | I | -- | -- | LM FT | A (ICML) |
| Biderman et al. (2024) <!--ref:arxiv:2405.09673--><!--anchor:none:--> | -- | -- | S | I | -- | LM CPT/IFT | A (TMLR) |
| Thinking Machines "LoRA Without Regret" <!--ref:web:thinking-machines-lora--><!--anchor:none:--> | -- | -- | S/C | -- | -- | industry blog | B |
| sayakpaul SVD gist <!--ref:web:sayakpaul-gist--><!--anchor:section:svd_low_rank_lora.py--> | -- | -- | S | -- | -- | tool | B |
| PERK (2025) <!--ref:arxiv:2507.06415--><!--anchor:none:--> | -- | I | S/C | -- | S | LM long-context | A (ICLR 2026 per agent) |
| FocuSFT (2026) <!--ref:arxiv:2605.09932--><!--anchor:none:--> | I | I | C | -- | S | LM | B |
| In-Place TTT (2026) <!--ref:arxiv:2604.06169--><!--anchor:none:--> / TTT-NTP (2026) <!--ref:arxiv:2606.21803--><!--anchor:none:--> | I | -- | -- | -- | S | LM | A / B |
| MASS (2026) <!--ref:arxiv:2603.03524--><!--anchor:none:--> / GradMem (2026) <!--ref:arxiv:2603.13875--><!--anchor:none:--> | I | -- | -- | -- | I | LM | B |
| Provable meta-LoRA, Block et al. (2024) <!--ref:arxiv:2410.22264--><!--anchor:none:--> | -- | -- | I | -- | S | theory | B |
| RW-TTT (2026) <!--ref:arxiv:2605.28053--><!--anchor:none:--> | -- | -- | -- | -- | I | serving system | B (not relevant to training memory) |

Peripheral, verified but not load-bearing: EMG MAML+LoRA (2601.04181), SEAL (2506.10943), AutoLoRA (2403.09113), MetaLoRA, WAM-TTT (2607.06988), Nested Learning (2512.24695), Tri Dao's Gram-NS blog.

### Key Themes

#### T1. Memory of checkpointing-through-time with large fast weights
**Evidence strength**: Strong on the formula and on the JAX path (code read directly: e2e, ttt-lm-jax, PyTorch-MAML, LaCT); Moderate on host offload (docs only, platform-dependent); Emerging/negative on MixFlow-MG and reversible updates.

**The cost model.** Two independent JAX codebases implement the same two-level structure — an outer `scan` over groups of n chunks, each group under `remat` — for exactly this reason: Sun et al. (2024) <!--ref:arxiv:2407.04620--><!--anchor:quote:we%20still%20need%20to%20save%20T%2Fb%20W_s%20at%20the%20end%20of%20the%20mini-batches--> and e2e's `scan_remat_chunk(inner_remat_freq)` <!--ref:web:e2e-repo--><!--anchor:section:jax_utils.py::scan_remat_chunk-->. The memory agent's reading of e2e gives peak per sequence ≈ (N/n + n)·(|W|+|v|) + A_step, times the vmapped local batch; with the default n=1 this is N full copies, and n≈√N gives ≈2√N. The TTT-E2E paper states the same scaling in words ("increase gradient checkpointing through time by a factor of log(T)") <!--ref:arxiv:2512.23675--><!--anchor:none:-->. PyTorch-MAML <!--ref:web:pytorch-maml--><!--anchor:quote:up%20to%2080%25%20GPU%20memory%20with%20~20%25%20more%20time--> is the PyTorch analogue with group size 1 and no √N grouping; LaCT does *no* checkpointing at all and tolerates O(N·|W|) only because its 2K–4K chunks give N=16 at 32K <!--ref:arxiv:2505.23884--><!--anchor:none:-->. Anchor: three code readings converge on the same formula; the paper's log(T) remark is consistent with a treeverse-depth reading.

**Planning table** (fp32 carry per sequence, per-device batch 1 via `accum_steps`, n = round(√N): N=8→n=3 (≈5.7 copies), N=32→n=6 (≈11.3 copies); SGD |v|=0; AdamW |v|=2|W| → 3×). 125M numbers are the DA's (4.72M MLP params/block, all-blocks 56.6M = 0.23 GB; last-1/4 14.2M). The 760M/32K figure (≈20 GB SGD / 62 GB AdamW) is pre-registered in the brief <!--ref:arxiv:2512.23675--><!--anchor:none:-->; it implies ≈1.77 GB per fp32 copy (≈440M fast params, ≈58% of the model), and the 350M row is **interpolated from that fraction and is an estimate only** (the corpus gives no 350M config).

| Model / fast set | |W| fp32 | 8K (N=8) SGD | 8K AdamW | 32K (N=32) SGD | 32K AdamW |
|---|---|---|---|---|---|---|
| 125M, last 1/4 | 0.06 GB | 0.3 GB | 1.0 GB | 0.6 GB | 1.9 GB |
| 125M, all blocks | 0.23 GB | 1.3 GB | 3.9 GB | 2.6 GB | 7.7 GB |
| 350M, all blocks (est.) | ≈0.8 GB | ≈4.6 GB | ≈14 GB | ≈9 GB | ≈27 GB |
| 760M, all blocks | ≈1.77 GB | ≈10 GB | ≈30 GB | ≈20 GB | ≈62 GB |

Add ≥3–4 live |W| buffers for the adjoint and HVP temporaries plus one chunk's activations (memory agent), and the vmapped batch multiplier if `accum_steps` is not used. Conclusion (C-002): the sub-question is arithmetic at 125M and only binds for 760M/32K/all-blocks with AdamW, exactly as the DA argued.

**What each strategy buys.**
- *√N grouping* (`inner_remat_freq≈√N`): N→2√N copies, zero new code, ≈1 extra forward per group. First choice.
- *Host offload* of group-boundary W via `checkpoint_name` + `save_and_offload_only_these_names(..., "pinned_host")` <!--ref:web:jax-memory-spaces--><!--anchor:quote:memory-kind%20support%20varies%20by%20platform-->: device holds ≈n copies, host N/n. e2e already imports `jax.ad_checkpoint` and dispatches named policies, so the patch is local; but the docs warn parameter offloading "works only when scanning over axis 0" and support is platform-dependent — verify with `Compiled.memory_analysis()` before relying on it. Second choice.
- *Treeverse* (`eqx.internal.scan(kind="checkpointed")` or a third scan level): O(log N) copies at O(log N)× recompute. Only if offload is unavailable; more recompute than √N grouping for N ≤ 32.
- *bf16 ΔW*: halves every copy but the DA's numerics note applies — a unit-norm global step over 56.6M elements moves each by ≈1e-4, comparable to bf16 resolution of a |W0|≈0.02 element; no TTT paper reports bf16 fast weights (LaCT and e2e keep fp32 `state_dtype`) <!--ref:arxiv:2505.23884--><!--anchor:none:-->. Keep W0 fp32, form W0+ΔW in fp32; last resort.
- *MixFlow-MG* <!--ref:arxiv:2505.00793--><!--anchor:none:-->: its formula is O(|A| + T·(|θ|+|v|)); the mixed-mode HVP trick shrinks |A|, i.e. the second-order activation term, at T∈{2..8} Adam steps. Our binding term is T·|W| with N=8–32 (128 at 128K), so its ~4× headline does not transfer; it also needs re-derivation per inner optimizer and has no repo (C-003). Not recommended.
- *Reversible updates* <!--ref:arxiv:1502.03492--><!--anchor:none:-->: exact reversal requires SGD-with-momentum and stored lost bits; clipped momentum-free SGD is non-invertible. Not applicable.
- *PyTorch compile / SAC / memory-budget API*: the AOT partitioner only sees one flat region and, decisively, `torch.compile` + `create_graph=True` is unsupported (pytorch #91469 open since 2022; builds raise "aot_autograd does not currently support double backward") <!--ref:web:pytorch-issue-91469--><!--anchor:none:-->. Usable only for the frozen prefix. Axolotl's offloading is `saved_tensors_hooks`/`save_on_cpu` tied to HF layer structure and first-order FT <!--ref:web:axolotl-docs--><!--anchor:none:-->.
- *Muon/AdamW inner*: Adam adds |v|=2|W| to every copy; Muon without momentum is stateless but its NS intermediates (5×(2a²+ab)) are saved or recomputed per step.

**Recommendation (T1).** Stay in JAX e2e: `inner_remat_freq = round(√N)` (3 at 8K, 6 at 32K), `accum_steps` so the vmapped batch is 1, stateless inner optimizer, fp32 W. If the 760M/32K arm still overruns 80 GB, add pinned-host offload of the `fast_w` carry; use `n_state_parallel` before bf16 ΔW. A PyTorch port would have to mirror this in eager with `torch.utils.checkpoint(use_reentrant=False)` per group and no compile in the meta-gradient region — strictly more error-prone. Evidence: convergent code readings (Level VI, three implementations) + vendor docs; the offload path is unmeasured.

#### T2. Inner optimizer under differentiate-through
**Evidence strength**: Strong (convergent) that stateless SGD-family updates are what every working system uses; Moderate that AdamW is ruled out (code comments + analytic derivation, no benchmark); Emerging/contested on Muon.

Every differentiate-through TTT system in the corpus — TTT-Linear/MLP <!--ref:arxiv:2407.04620--><!--anchor:none:-->, TTT-E2E (`clip_by_global_norm(1.0)` + `sgd(lr=1)`) <!--ref:web:e2e-repo--><!--anchor:none:-->, LaCT (Eq. 8 `L2-Normalize(W − g)`) <!--ref:arxiv:2505.23884--><!--anchor:section:Eq.8-->, E²-TTT (GD + learned lr and weight decay) <!--ref:arxiv:2608.21308--><!--anchor:none:-->, Titans/Atlas (GD with data-dependent momentum and decay) <!--ref:arxiv:2501.00663--><!--anchor:none:--> — uses an SGD-family rule and none uses Adam. The meta-learning literature explains why: Metz et al. show unrolled-optimization gradients are "either strongly biased (short truncations) or exploding norm (long truncations)" <!--ref:arxiv:1810.10180--><!--anchor:none:-->; `higher` warns adaptive differentiable optimizers "may be unstable and yield NaNs when taking higher order gradients" and masks `exp_avg_sq==0` in a backward hook <!--ref:web:higher--><!--anchor:section:higher/optim.py-->; the optimizer agent's derivation gives dΔ/dg = ε/(|g|+ε)² at step 1, i.e. ≈0 for |g|≫ε and ≈1e8 for |g|≪ε at ε=1e-8. Combined with the T1 memory cost (2|W| state per copy, 62 GB at 760M/32K), AdamW ε=1e-8 is ruled out (C-005); raising ε to ≈1e-3 yields a 3×-memory, worse-conditioned normalized SGD. The one dissent is PERK, which differentiates through 4 AdamW steps via `higher` — but on a rank-256 LoRA with truncated unrolling of the last 1–2 steps <!--ref:arxiv:2507.06415--><!--anchor:none:-->, a regime that does not transfer to 8–32 full-rank steps.

*Global vs per-tensor normalization.* The paper's rule is a single global norm over all fast params, capping ‖ΔW_all‖_F ≤ 1 per chunk (≈2% relative change on a 1536×3328 matrix at init std 0.02, per the optimizer agent). Spreading a fixed global step over 4–5× more parameters shrinks the per-tensor step, so the DA's "bigger fast weights store no more" self-inflicted-null risk is real; the brief already moves the sweep to a per-tensor RMS unit at {1, 3, 10}×. Per-tensor normalization with a learned scalar per tensor (LaCT's `softplus(Linear(x)+inv_softplus(base_lr))`, MAML++ LSLR, Meta-SGD) <!--ref:arxiv:1810.09502--><!--anchor:none:--> is stateless, differentiable, adds only outer params, and is the same "learned per-tensor inner LR" the DA asked to put in the slow set. Backward of g/‖g‖ is (I−ĝĝᵀ)/‖g‖: well conditioned when ‖g‖≫ε.

*Muon.* LaCT reports "Muon's surprising effectiveness over Vanilla Gradient Descent and Momentum" (Fig. 7b, no numbers in text) and that "Muon also improves the numerical stability in our setup" <!--ref:arxiv:2505.23884--><!--anchor:quote:Muon%20also%20improves%20the%20numerical%20stability%20in%20our%20setup-->; yet its LLM code default is `use_muon=False` — Muon is the ablation, not the recipe. Atlas's Table 6 shows 19.97 ppl with Muon vs 19.65 without <!--ref:arxiv:2505.23735--><!--anchor:section:Table%206-->, i.e. no gain. Resolution: LaCT's "vanilla GD" baseline still includes row-L2 weight renorm and its chunks are 2K–1M tokens with square d×d weights; Atlas uses momentum + decay and a sliding-window loss. Muon's value appears where the raw chunk gradient is high-rank enough for orthogonalization to help; on a single 1024-token chunk the gradient is low-rank and noisy, so orthogonalizing it to a full-spectrum update is a hypothesis (DA minor note), not a default. Cost is the decisive practical point: NS5 ≈ 20a²b+10a³ FLOPs per matrix per chunk; the overhead relative to the matrix's own fwd+bwd scales as ≈(10a/B + 5a²/(bB))/3 — Keller's own formula gives ≈750% at B=1024 tokens/step vs 0.5–0.7% in pretraining <!--ref:web:muon-blog--><!--anchor:none:-->. For 125M (768×2048) that is ≈3× the matrix's fwd+bwd; for LaCT-size 1536×4096 ≈6×. Backward through NS5 has slope ≈485–493 at zero singular value in the agent's analysis (no paper found on double-backward through NS), the most ill-conditioned of the candidates. If used: Moonshot's 0.2·√max(A,B) RMS scaling <!--ref:arxiv:2502.16982--><!--anchor:section:Lemma%201--> so the LR unit matches the normalized-SGD sweep; no momentum (stateless).

**Recommendation (T2).** Default: per-tensor normalized SGD with a learned per-tensor inner-LR scalar in the slow set (RMS unit, sweep {1,3,10}× paper-equivalent), no state, fp32. Keep the paper's global-norm rule as the reference arm. Run Muon-no-momentum as one ablation at 125M only, with RMS-matched scale and a detach-able NS (LaCT: "adding detach here sometimes improves stability"). Do not run AdamW inner except as a single ε=1e-3 sanity point if budget allows. Evidence: convergence of five TTT systems + two meta-learning analyses (Level VI, grade A/B); Muon contested (1 for, 1 against).

#### T3. Slow LoRA: placement, rank, scaling, outer LR
**Evidence strength**: Moderate. Rank/scaling evidence is from fine-tuning, not meta-gradients; the only meta-gradient LoRA (PERK) uses a frozen base and LoRA fast weights.

*Where the LoRA is redundant (DA issue 1).* With LoRA on the same MLP that is the fast weight, the forward is (W_i + BA)x ≡ W_i x with W0 := W_pre + BA; the inner loop is unchanged and the outer loop is restricted to a rank-r shift of the fast init. Block et al. formalize the dual: meta-learn a base so rank-r adapters adapt well <!--ref:arxiv:2410.22264--><!--anchor:none:-->, which is arm C's MLP-only ablation viewed from the other side. The paper's finding that all-slow meta-learning is what separates TTT-E2E from TTT-naive <!--ref:arxiv:2512.23675--><!--anchor:none:--> is what the MLP-only LoRA cannot reach. The brief's revised default slow set — attention Q/K/V/O LoRA, MLP LoRA, all RMSNorm gains, per-tensor inner LRs — is supported by three convergent lines: LaCT's outer loop learns per-token per-matrix lrs and the fast init <!--ref:arxiv:2505.23884--><!--anchor:none:-->, PERK learns per-layer-per-step inner LRs <!--ref:arxiv:2507.06415--><!--anchor:none:-->, MAML++/Meta-SGD learn per-layer/per-parameter LRs <!--ref:arxiv:1707.09835--><!--anchor:none:--> (C-007). The one apparent contradiction — Thinking Machines: "Attention-only LoRA significantly underperforms MLP-only LoRA" <!--ref:web:thinking-machines-lora--><!--anchor:quote:Attention-only%20LoRA%20significantly%20underperforms%20MLP-only%20LoRA--> versus Hu et al.'s {Wq,Wv} r=8 preference <!--ref:arxiv:2106.09685--><!--anchor:section:Table%205--> — is about fine-tuning a frozen model, where the MLP is where capacity lives. Here the MLP is *already* fully trainable in the inner loop, so the SFT placement result does not transfer; the attention LoRA's job is different (deciding what gets written into fast memory). Verdict: MLP LoRA stays as ablation; attention LoRA + norms + inner LRs are the arm.

*Rank and scaling.* Hu et al.'s Table 6 (r=1..64 flat) is an SFT result. Biderman et al. show full-FT ΔW rank is 10–100× typical LoRA ranks, grows with data, is higher for MLP than attention, and LoRA underperforms full FT for CPT even at r=256 / 20B tokens <!--ref:arxiv:2405.09673--><!--anchor:none:-->; the outer objective here (125M–500M DCLM tokens of meta-training) is CPT-scale, so the sweep must reach 256, not stop at 128 as the blueprint has it. rsLoRA's Thm 3.2 requires Θ(1/√r) scaling to avoid gradient collapse at large r <!--ref:arxiv:2312.03732--><!--anchor:section:Thm%203.2-->; Thinking Machines' 1/r finding (LR rank-independent, identical early curves) is consistent for r ≤ 64 but was not tested at 256. PERK used rsLoRA with α→256 at r=256 <!--ref:arxiv:2507.06415--><!--anchor:none:-->. Convergent recommendation: r ∈ {16, 64, 256}, rsLoRA α/√r with fixed α; B=0 init so the outer loop starts exactly at the TTT-naive point (arm B), which makes C-vs-B a clean ablation. The sayakpaul gist <!--ref:web:sayakpaul-gist--><!--anchor:section:svd_low_rank_lora.py--> (ΔW=BA → randomized SVD → B_new=U√S, A_new=√S·Vh, per-module relative Frobenius error) is a post-hoc rank reader: train at r=256, truncate to k∈{4..64}, evaluate meta-test loss per k without retraining. Caveats from the agent: it ignores `scaling` (fold α/√r into B before truncation), hardcodes CUDA (port to NumPy/JAX), and a truncated high-rank adapter is not the optimum at low rank — the curve is diagnostic, not a substitute for the sweep.

*Outer LR (tension).* Two anchors disagree by two orders of magnitude. The 10× rule (Thinking Machines multiplier 9.8; Biderman "one order of magnitude higher than full finetuning, often 5e-5 to 5e-4"; Hu et al. 2e-4 vs 5e-6) applied to TTT-E2E's full-weight LRs (3e-3 pretrain at 125M, 4e-4 extension) gives 4e-3 to 3e-2. PERK and FocuSFT used 1e-5 <!--ref:arxiv:2605.09932--><!--anchor:none:-->. Resolution: PERK/FocuSFT are 0.5B–7B models fine-tuned from instruction-tuned checkpoints with Adam at conventional 7B LRs (full-FT 1e-5 for FocuSFT), so 1e-5 is the *full-FT* LR at that scale, not a LoRA rule; the 10× rule applied to FocuSFT's 1e-5 would itself give 1e-4. The relevant full-FT anchor at 125M is the paper's 4e-4 extension LR (arm C is an extension-style run), giving a centre of ≈3e-3 and a bracket {3e-4, 1e-3, 3e-3, 1e-2} at 125M, {1e-4 … 3e-3} at 760M, keeping β=(0.9,0.95), wd 0.1, clip 1.0, 10% warmup, cosine→1e-5. Second-order gradients through 8–32 steps have no LoRA-specific LR evidence at all; the bracket must be swept, not assumed (C-008). *Batch sensitivity*: Thinking Machines report LoRA is less tolerant of large batches than full FT, independent of rank; the paper doubles batch for extension — prefer the smaller pretraining batch at the higher LR. LoRA+ (η_B/η_A=16) is an optional 1–2% lever; DoRA's magnitude parameter has no motivation here.

**Recommendation (T3).** Slow set = attention LoRA (Q/K/V/O, half rank per matrix at equal budget) + MLP LoRA (ablation) + RMSNorm gains + per-tensor inner LR scalars; r ∈ {16,64,256}, rsLoRA, α fixed, B=0 init; outer LR bracket as above; SVD-truncation curve at the best rank. Evidence: fine-tuning literature (grade A/B, convergent on 10× and rsLoRA) extrapolated to meta-gradients (no direct evidence — Gap 2).

#### T4. Forgetting and stability of pretrained MLPs as fast weights
**Evidence strength**: Emerging. No source measures knowledge retention of large fast weights; stabilizers exist but were chosen for TTT quality, not for forgetting.

The paper's design argument is structural: the original MLP stays static as "safe storage" and a second `feed_forward_prime` MLP receives all inner steps <!--ref:arxiv:2512.23675--><!--anchor:none:-->; the brief's Fig. 4-right reading (more fast *layers* helps context scaling) is about prime-MLP depth, not about overwriting pretrained matrices. Arm C removes safe storage entirely. LaCT overwrites 40% of parameters ("state-to-parameter size ratio ≥40%") with a learnable init and a *row-wise L2 renorm* of W after every update, justified as "analogous to the post-layer norm" and against "magnitude explosion or decayed memory" <!--ref:arxiv:2505.23884--><!--anchor:quote:Fast-weight%20updates%20in%20TTT%20repeatedly%20accumulate%20gradients%2C%20and%20thus%20suffer%20from%20magnitude%20explosion%20or%20decayed%20memory-->. Titans/Atlas keep an explicit data-dependent decay (1−α_t)M_{t−1} <!--ref:arxiv:2501.00663--><!--anchor:none:-->, Atlas's ablation shows the window/polynomial features matter more than Muon <!--ref:arxiv:2505.23735--><!--anchor:section:Table%206-->; E²-TTT learns a weight decay (0.1). In-Place TTT and TTT-NTP <!--ref:arxiv:2606.21803--><!--anchor:none:--> confine writes to the MLP down-projection (rank-one accumulated writes; W_i = W0 + ΣΔ_j), which is a *structural* bound on damage rather than a regularizer. Apparent contradiction "safe storage vs normalization-only": LaCT's fast weights are *meta-learned from scratch* and never held pretrained knowledge, so its renorm protects TTT dynamics, not retention; the paper's safe storage protects pretrained knowledge. They answer different questions and are compatible; neither resolves whether a pretrained MLP under 8–32 unit-norm steps retains general knowledge (C-009). Biderman's "LoRA forgets less" is about outer fine-tuning of a frozen base and is a further reason the *slow* LoRA is safe, not the fast MLP. The DA's argument that a fixed global step concentrated on a few rows can do local damage invisible to mean loss stands unrebutted by any source.

**Recommendation (T4).** Keep the brief's forgetting probe (ΔNLL of W_T on a fresh DCLM chunk vs W0) and λ-decay sweep {0, 0.05, 0.2}; add LaCT-style row renorm of W (with `w_norm` = the pretrained row norm, so W0 is a fixed point) as a second stabilizer and In-Place-style "w2-only fast" as a structural ablation; run the DA's safe-storage-preserving variant (cloned MLP as fast) at least once — it is also arm F. Evidence: Emerging (structural arguments from 4 sources, no measurement).

#### T5. Precedents and novelty
**Evidence strength**: Strong (a comprehensive Phase 2 scan found no exact match; convergent on the nearest neighbours).

What exists: PERK — meta-learned LoRA init as slow, LoRA as fast, frozen base, 4 Adam steps, 8K→128K extrapolation <!--ref:arxiv:2507.06415--><!--anchor:none:--> (closest in outer structure, opposite in fast structure); FocuSFT — fast LoRA r=32 on FFN of top 35% layers, K=2, inner lr 1.0 clip 1.0, slow = full 7B <!--ref:arxiv:2605.09932--><!--anchor:none:--> (the exact inverse, first-order); In-Place TTT / TTT-NTP — large fast weights in pretrained MLP down-projections at 0.5B–14B, but first-order, no meta-gradient, no LoRA <!--ref:arxiv:2604.06169--><!--anchor:none:-->; LaCT — 40% fast SwiGLU with everything slow, no LoRA, no pretrained init; TTT-E2E — small fast, all slow, from scratch; dynamic evaluation (Krause et al. 2018 — cited in the brief, not in the verified corpus) — arm B exactly. GradMem, MASS and WAM-TTT are second-order but with tiny fast state or ≤5 steps. What is new (C-010): (i) a *LoRA-only slow set meta-learned through the inner loop* with *full-rank pretrained MLPs as fast weights*; (ii) the merge-after-meta-training property (deployed model = pretrained Transformer + a per-chunk gradient step, the DA's strongest practical point); (iii) post-hoc TTT-ification at 5% of pretraining tokens, which is the paper's stated future direction but realized with LoRA rather than full-slow. What is not new: per-tensor learned inner LRs (Meta-SGD/MAML++/LaCT/PERK), full-weight dynamic evaluation (arm B), large fast MLPs (LaCT, In-Place TTT), meta-learned adapter inits (PERK, Block et al.). Novel 3/5 in the brief is therefore correct; the sharper framing the DA proposed — "which few slow parameters make dynamic evaluation of a pretrained Transformer meta-learnable?" — is the one no precedent addresses.

### Contradictions & Resolutions

| Claim A | Claim B | Resolution |
|---|---|---|
| LaCT: Muon clearly beats vanilla GD in fast-weight TTT (Fig. 7b) | Atlas Table 6: removing Muon improves ppl 19.97→19.65 | Reconcilable: different baselines (LaCT's GD keeps row renorm; Atlas has momentum+decay+windowed loss), chunk sizes (2K–1M vs token-level), weight shapes. Muon's gain is conditional on high-rank chunk gradients; at b=1024 it is a hypothesis. ¶T2. |
| Thinking Machines / Biderman / Hu: LoRA LR ≈ 10× full-FT LR | PERK, FocuSFT: outer LR 1e-5 | Reconcilable: 1e-5 is the full-FT-scale LR of 0.5B–7B instruction-tuned models, not a LoRA rule; the correct anchor at 125M is the paper's 4e-4 extension LR → bracket 3e-4…1e-2. No evidence for either under second-order gradients. ¶T3. |
| TTT-E2E: static MLP as safe storage is needed | LaCT: 40% of weights fast with only row-L2 renorm | Reconcilable but non-transferable: LaCT's fast weights never held pretrained knowledge; the two protect different things. Leaves forgetting of pretrained fast weights untested. ¶T4. |
| MixFlow-MG: ~4× memory reduction for meta-gradients | Memory agent: negligible benefit for our T·|W| term | Reconcilable: MixFlow-MG's O(|A| + T(|θ|+|v|)) attacks |A| at T≤8 Adam steps; our binding term is T·|W| at N=8–32 with |v|=0. Not contradictory, different regime. ¶T1. |
| Thinking Machines: attention-only LoRA underperforms MLP-only | Hu et al.: {Wq,Wv} r=8 best, MLP never adapted; DA: MLP LoRA is redundant with W0 here | Reconcilable: both are frozen-base SFT results; with the MLP trainable in the inner loop, MLP LoRA collapses into W0 and attention LoRA serves a different role. ¶T3. |
| Hu et al. Table 6: rank 1–64 flat | Biderman: full-FT ΔW rank 10–100× LoRA ranks, LoRA < full FT at CPT scale | Reconcilable: SFT vs CPT data scale; outer objective here is CPT-scale → sweep to 256 with rsLoRA. ¶T3. |
| Keller Jordan: Muon overhead 0.5–0.7% | Optimizer agent: ≈750% overhead at B=1024 | Not a contradiction: same formula T·m/B evaluated at pretraining batch (0.5–16M tokens) vs TTT chunk (1024 tokens). ¶T2. |

#### Cross-Paper Tension Inventory (#262)

```yaml
cross_paper_tensions:
  - pair_id: CP-001
    paper_a: "arxiv:2505.23884"
    paper_b: "arxiv:2505.23735"
    candidate_basis: "shared construct (Muon/Newton-Schulz inside a TTT inner loop); opposite finding direction"
    overlap_topic: "Does orthogonalizing the fast-weight gradient (Muon) improve TTT quality?"
    a_finding: "Muon 'surprisingly effective' over vanilla GD and momentum (Fig. 7b, no numbers in text); LLM code default use_muon=False"
    a_evidence_pointer: "search_layer_inner_optimizer.md §1 (LaCT Fig. 7(b), code defaults)"
    b_finding: "Ablation: Atlas 19.97 ppl with Muon vs 19.65 'w/o Muon'; polynomial features and window c>1 matter more"
    b_evidence_pointer: "search_layer_inner_optimizer.md §3 (Atlas Table 6)"
    pair_assessment: "conditional_difference"
    resolution_status: "resolved_in_synthesis"
    resolution_pointer: "Synthesis Report > Contradictions & Resolutions, row 1; Key Themes T2 ¶Muon"
    scholar_confirmation: "pending"
  - pair_id: CP-002
    paper_a: "web:thinking-machines-lora"
    paper_b: "arxiv:2507.06415"
    candidate_basis: "shared construct (LoRA outer learning rate); opposite finding direction"
    overlap_topic: "What outer/AdamW learning rate should a LoRA use?"
    a_finding: "Optimal LoRA LR ≈ 10× full-FT LR (multiplier 9.8), rank-independent under 1/r scaling; Biderman (2405.09673) concurs: 5e-5–5e-4"
    a_evidence_pointer: "search_layer_lora.md §3, §5"
    b_finding: "PERK meta-learns LoRA init with outer AdamW lr 1e-5; FocuSFT (2605.09932) slow full-7B AdamW 1e-5"
    b_evidence_pointer: "search_layer_lora.md §6"
    pair_assessment: "conditional_difference"
    resolution_status: "resolved_in_synthesis"
    resolution_pointer: "Synthesis Report > Contradictions & Resolutions, row 2; Key Themes T3 ¶Outer LR"
    scholar_confirmation: "pending"
  - pair_id: CP-003
    paper_a: "arxiv:2512.23675"
    paper_b: "arxiv:2505.23884"
    candidate_basis: "shared RQ subtopic (stability of large fast weights); opposite design choice"
    overlap_topic: "Is a static 'safe storage' MLP necessary, or does weight normalization suffice for large fast weights?"
    a_finding: "Original MLP kept static; a separate prime MLP receives inner steps so pretrained knowledge cannot be erased"
    a_evidence_pointer: "research_question_brief.md baseline-facts table (Fast weights row); da_checkpoint1.md issue 3"
    b_finding: "SwiGLU fast weights up to 40% of params with learnable init, stabilized only by channel-wise L2 renorm ('like post-norm')"
    b_evidence_pointer: "search_layer_inner_optimizer.md §1 (normalization rationale quotes); search_layer_memory_checkpointing.md §6b"
    pair_assessment: "conditional_difference"
    resolution_status: "resolved_in_synthesis"
    resolution_pointer: "Synthesis Report > Contradictions & Resolutions, row 3; Key Themes T4"
    scholar_confirmation: "pending"
  - pair_id: CP-004
    paper_a: "arxiv:2505.00793"
    paper_b: "web:e2e-repo"
    candidate_basis: "shared construct (meta-gradient memory); agent-noted cross-cluster"
    overlap_topic: "Which memory term dominates differentiate-through TTT, and does MixFlow-MG reduce it?"
    a_finding: "~4× less memory (80% of configs), >10× peak, by mixed-mode HVPs; formula O(|A| + T(|θ|+|v|)), T∈{2..8} Adam steps"
    a_evidence_pointer: "search_layer_memory_checkpointing.md §2"
    b_finding: "e2e peak ≈ (N/n + n)(|W|+|v|) + A_step; with N=8–128 and large W the carry term dominates"
    b_evidence_pointer: "search_layer_memory_checkpointing.md 'What e2e costs and the levers'"
    pair_assessment: "no_material_conflict"
    resolution_status: "not_applicable"
    scholar_confirmation: "pending"
  - pair_id: CP-005
    paper_a: "web:thinking-machines-lora"
    paper_b: "arxiv:2106.09685"
    candidate_basis: "shared construct (LoRA placement); opposite finding direction"
    overlap_topic: "Attention vs MLP placement of LoRA"
    a_finding: "Attention-only LoRA significantly underperforms MLP-only; apply to all layers, especially MLP"
    a_evidence_pointer: "search_layer_lora.md §3"
    b_finding: "GPT-3 175B Table 5: {Wq,Wv} r=8 best; MLP never adapted"
    b_evidence_pointer: "search_layer_lora.md §1"
    pair_assessment: "conditional_difference"
    resolution_status: "resolved_in_synthesis"
    resolution_pointer: "Synthesis Report > Contradictions & Resolutions, row 5; Key Themes T3 ¶Where the LoRA is redundant"
    scholar_confirmation: "pending"
  - pair_id: CP-006
    paper_a: "arxiv:2106.09685"
    paper_b: "arxiv:2405.09673"
    candidate_basis: "shared construct (LoRA rank sufficiency)"
    overlap_topic: "Does rank matter beyond ~8–64?"
    a_finding: "Table 6: r=1..64 essentially flat for SFT"
    a_evidence_pointer: "search_layer_lora.md §1"
    b_finding: "LoRA underperforms full FT for CPT even at r=256/20B tokens; full-FT ΔW rank grows with data, MLP > attention"
    b_evidence_pointer: "search_layer_lora.md §5"
    pair_assessment: "conditional_difference"
    resolution_status: "resolved_in_synthesis"
    resolution_pointer: "Synthesis Report > Contradictions & Resolutions, row 6; Key Themes T3 ¶Rank and scaling"
    scholar_confirmation: "pending"
  - pair_id: CP-007
    paper_a: "web:muon-blog"
    paper_b: "arxiv:2505.23884"
    candidate_basis: "shared construct (Newton-Schulz cost)"
    overlap_topic: "Is Muon's Newton-Schulz overhead negligible?"
    a_finding: "Overhead T·m/B → 0.5–0.7% in pretraining"
    a_evidence_pointer: "search_layer_inner_optimizer.md §2"
    b_finding: "NS5 '30·b·d³ FLOPs' per matrix per chunk, amortized by 2K–1M-token chunks"
    b_evidence_pointer: "search_layer_memory_checkpointing.md §6b; search_layer_inner_optimizer.md §2"
    pair_assessment: "no_material_conflict"
    resolution_status: "not_applicable"
    scholar_confirmation: "pending"
  - pair_id: CP-008
    paper_a: "arxiv:2507.06415"
    paper_b: "web:higher"
    candidate_basis: "shared construct (differentiable Adam in an inner loop)"
    overlap_topic: "Can Adam be differentiated through safely?"
    a_finding: "PERK differentiates through 4 AdamW steps via higher with truncated unrolling (last 1–2 steps), learned per-layer-per-step LRs"
    a_evidence_pointer: "search_layer_lora.md §6"
    b_finding: "higher: adaptive differentiable optimizers 'may be unstable and yield NaNs'; masks exp_avg_sq==0 in a backward hook"
    b_evidence_pointer: "search_layer_inner_optimizer.md §4"
    pair_assessment: "conditional_difference"
    resolution_status: "flagged_unresolved"
    scholar_confirmation: "pending"
```

**Coverage Note**: 33 arXiv + 23 web/repo sources in corpus (≈27 load-bearing); 8 candidate pairs considered (basis: shared construct / opposite finding direction / shared RQ subtopic / agent-noted cross-cluster). This is a **scoped advisory scan, not complete pairwise contradiction detection** — cross-neighborhood pairs not surfaced here may exist and are not claimed absent (in particular, pairs among the six 2026 first-order TTT papers were not checked against each other). Bibliographic coupling was used as an inclusion signal only. CP-008 is left unresolved because no source benchmarks differentiable Adam at 8–32 full-rank steps. Scholar confirms each `resolution_pointer` and may flag additional cross-pairs.

### Knowledge Gaps

1. **Empirical — forgetting of pretrained fast weights.** No source evaluates knowledge retention after TTT overwrites pretrained MLPs (In-Place TTT / TTT-NTP / LaCT report task loss only). Implication: the brief's forgetting probe is the first such measurement; it must be in the headline table.
2. **Methodological — LoRA hyperparameters under meta-gradients.** All rank/scaling/LR evidence is from first-order fine-tuning of frozen bases; PERK is the only meta-gradient LoRA and it uses a frozen base, LoRA fast weights and 4 truncated Adam steps. Implication: the outer-LR and rank brackets must be swept, and the SVD effective-rank curve reported, because no prior is trustworthy.
3. **Methodological — inner-optimizer benchmark inside unrolled TTT.** No paper compares SGD vs normalized SGD vs Muon vs Adam inside a MAML-style loop at LM scale; Muon's double-backward conditioning has no published analysis (agent derivation only). Implication: the {global, per-tensor, Muon} sweep at 125M is itself a contribution; report inner-gradient norm statistics.
4. **Empirical — host offload of scan carries in JAX.** Documented API, no TTT paper or measurement; platform support "varies". Implication: measure with `Compiled.memory_analysis()` before the 760M/32K arm is scheduled.
5. **Empirical/temporal — 350M numbers.** The corpus has no 350M config; the planning row is interpolated. Implication: fill in from the e2e config before the scale check.
6. **Theoretical — what a low-rank slow set can meta-learn.** Block et al. give the reverse guarantee (meta-learn base for LoRA adapters); no theory covers LoRA-slow / full-rank-fast. Implication: H1 is a purely empirical hypothesis.

### Evidence Convergence Map

```
Strong:      [==========] T1 memory formula & JAX scan+remat path   (5 code/doc sources, convergent)
Strong:      [=========-] T2 stateless SGD-family inner update      (5 TTT systems + 2 analyses)
Strong:      [=========-] T5 nearest precedents / no exact match    (8 sources, Phase 2 scan)
Moderate:    [======----] T3 rank/scaling (rsLoRA, r→256)           (4 fine-tuning sources, extrapolated)
Moderate:    [=====-----] T3 outer LR bracket (10× rule)            (3 sources for, 2 apparent against, resolved)
Moderate:    [=====-----] T1 host offload                            (docs only)
Emerging:    [===-------] T2 Muon inside TTT at b=1024              (1 for, 1 against, cost analysis)
Emerging:    [==--------] T4 forgetting / stabilizers                (structural arguments, 0 measurements)
Gap:         [----------] LoRA hyperparameters under meta-gradients; 350M config; Adam-in-loop benchmark
```

### Theoretical Integration

The findings sit inside the bilevel (MAML-style) framing the paper uses: inner loop = few normalized gradient steps on next-token loss; outer loop = differentiate through them. Three theoretical threads from the corpus organize the recommendations. (1) *Unrolled-gradient pathology* (Metz; Andrychowicz; `higher`): the outer gradient is only as well-conditioned as the inner update's Jacobian, which favours updates whose backward is a projection (normalized SGD) over those with ε-singularities (Adam) or high polynomial slope (NS5). (2) *Learned preconditioning as slow capacity* (Meta-SGD, MAML++, LaCT, PERK): a stateless learned per-tensor LR is the cheapest place to put meta-learnable capacity and makes the "small slow set" hypothesis H1 coherent — it is where the paper's all-slow outer loop plausibly spends much of its leverage. (3) *Rank of the outer shift* (rsLoRA, Biderman, Block et al.): the outer objective is CPT-scale, so the rank needed to move the init is unknown and probably larger than SFT priors; the theory of Block et al. runs the other way and gives no bound. The safe-storage question (T4) has no framework in the corpus at all — the closest is LaCT's post-norm analogy — and is the theme most likely to produce a surprise.

### Synthesis Limitations

- All sources are Level VI single studies or code; "strength" here is convergence across implementations, not hierarchy level. Six load-bearing sources are 2026 unrefereed preprints (grade B).
- Numerical claims (Muon FLOPs, Adam derivative, bf16 resolution, memory copies) are the Phase 2 agents' derivations or readings of code; the verification report states they were not independently re-read. The 350M memory row is an interpolation by this agent.
- The corpus was assembled around the meta-gradient reading of the user's intent (brief Q2) and per-sequence reset (Q1); if the user chooses consolidation or persistence, T3 and T4 need a different literature (continual learning, distillation) not searched here.
- Dynamic evaluation (Krause et al. 2018) is referenced from the brief only and is not a verified corpus source.
- Contradiction scan is scoped (8 pairs); no claim of completeness.
- Several anchors are `none` because the search-layer files quote sections/equations for only some sources; the finalizer will surface these.

**Recommendation to caller**: proceed to Phase 4 (report compilation) with this synthesis; the two open user questions (Q1 reset vs persist, Q2 meta-gradient vs consolidation) still gate implementation and should be surfaced in the report's front matter.
