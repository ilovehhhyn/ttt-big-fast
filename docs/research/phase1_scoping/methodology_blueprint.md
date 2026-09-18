## Methodology Blueprint

### Research Paradigm
**Selected**: Positivist / empirical ML (controlled ablation experiments with a fixed evaluation protocol).
**Justification**: The RQ is a comparative-performance and cost question; it is answered by measured held-out loss and measured GPU memory/latency under identical data, tokenizer and evaluation code.

### Method
**Type**: Quantitative, comparative.
**Specific Method**: Controlled ablation on the e2e codebase, reusing the paper's "basic recipe" (Appendix B) and its DCLM protocol (Section 3.1, 3.2). Five arms at 125M, matched total parameters and evaluation:
| Arm | Fast (inner) weights | Slow (outer) weights | Purpose |
|---|---|---|---|
| A. SWA baseline | none | none (frozen pretrained) | floor |
| B. TTT-naive | MLPs of chosen blocks | none (no meta-training) | isolates meta-learning contribution (= dynamic evaluation) |
| C. **Big-fast / LoRA-slow (proposed)** | MLPs (w1,w2,w3) of last 1/4, 1/2, or all blocks, W0 = pretrained | LoRA (A,B) on the same MLPs [+ optionally attention]; norms optionally slow | main arm |
| D. Big-fast / full-slow | same fast set | all params (paper-style) | upper bound on what the LoRA is giving up |
| E. Paper TTT-E2E | `feed_forward_prime` in last 1/4 | all params | reference point (released 125M checkpoint, or rerun) |

**Justification**: Arms A/B/E reproduce paper quantities so the new arm is anchored to published numbers (paper Table 1 / Fig. 4 give 760M numbers; 125M values will be regenerated).

### Data Strategy
**Data Type**: Secondary (public corpora, paper's tokenized buckets).
**Sources**: `gs://llama3-dclm-filter-8k` (DCLM-Baseline, docs ≥8K tokens, Llama-3 tokenized; train/val splits as zarr) for meta-training and evaluation at 8K; `gs://llama3-books3` for optional 32K extension. Both Requester-Pays.
**Sampling**: Meta-training tokens for arm C: start at 5% of the paper's Chinchilla pretraining budget (=125M tokens at 125M, i.e. the paper's *extension* budget: 120 steps × 32 seqs × 32K, or equivalently ~240 steps × 64 × 8K), since a LoRA + pretrained init should need far fewer tokens than from-scratch meta-training. Escalate to 20-50% if the loss is still improving.
**Time Frame**: Single-epoch, one pass, as in the paper.

### Analytical Framework
**Technique**: Held-out loss (mean CE on DCLM val at 8K; Books val at 32K if extended), per-token-index loss curves (paper Fig. 2/6 style, already produced by `Evaluator` → `token_nll_loss`), Δloss vs. arm A, plus measured peak GPU memory and sec/1K-tokens for training and prefill.
**Steps**:
1. Obtain pretrained SWA 125M: run `+experiment=125m/pretrain/pretrain-125m-fa` (at 8K, SWA k=8K ≡ full attention) — 4800 steps × 0.5M tokens ≈ 2.5B tokens, ~1.9e18 FLOPs (~3-6 H100-hours), or request the authors' baseline checkpoint.
2. Implement in e2e: (i) `LoRALinear` wrapping `NormalLinear` (B=0 init, A Gaussian, scaling α/r or rsLoRA α/√r); (ii) config knobs `fast_blocks`, `lora_rank`, `lora_targets`; (iii) `spec_inner` → base MLP weights of fast blocks, `spec_outer` → LoRA params (+ optional norms); (iv) loader that maps the pretrained checkpoint into the model with `load_part=params`; (v) inner optimizer options `normalized_sgd` (per-tensor or global) and `muon_nomomentum` (Newton-Schulz, RMS-matched scale); (vi) memory: `state_dtype` option for bf16 residual fast weights (store ΔW = W_fast − W0 in bf16, W0 in bf16 too), `inner_remat_freq` ≈ √(T/b), `accum_steps` so per-device vmap batch = 1.
3. Sweep (125M, 8K, fast = last 1/4 first): inner LR {0.3, 1, 3} × normalizer {global, per-tensor, muon} → pick best by val loss; then LoRA rank {8, 32, 128} × outer LR {1e-4, 3e-4, 1e-3, 3e-3}; then fast fraction {1/4, 1/2, all}.
4. Report table + per-token curves; measure memory with `XLA_PYTHON_CLIENT_ALLOCATOR=platform` peak stats / `jax.local_devices()[0].memory_stats()`.
5. Optional: 32K Books extension for the best config vs arms A/E.
**Tools**: JAX 0.5.x + Equinox + Optax (as in e2e), W&B, 1-8 × H100/H200 80GB (or GB200).

### Validity Criteria
| Criterion | Strategy to Ensure |
|-----------|-------------------|
| Internal validity (same data/eval) | Identical tokenized data, seeds, eval split and eval code across arms; paper's `Evaluator` reused unchanged |
| Matched cost | Report total params, inference FLOPs/token, training sec/1K tokens for every arm; arm E and C at equal total params |
| Reproducibility | Hydra configs committed per arm; W&B run ids logged in the results table |
| Statistical significance | Paper treats Δloss < 0.001 as noise; report at least 2 seeds for the headline arm |
| Construct validity of "memory" | Peak device memory measured, not estimated; the estimate table in Phase 3 is only for planning |

### Limitations (By Design)
- 125M-scale results may not transfer to ≥1B (paper Fig. 5 shows regime change ~760M); mitigated by one 350M/760M confirmation run of the best config.
- Starting from a pretrained *full-attention-at-8K* model means SWA ≡ full attention during meta-training; the SWA-vs-TTT gap only appears at ≥16K, so the 32K Books extension is needed to see context-scaling benefits.
- Training latency will exceed the paper's for "all blocks" (no prefix shortcut, XLA attention everywhere); measured and reported, not hidden.

### Ethical Considerations
- Public web/books corpora with known licensing controversy (Books3); use only for research replication as the paper did, and note it.

### Reporting Standard
- Recommended guideline: ML reproducibility checklist (NeurIPS-style); no human subjects.

### Preregistration
- Recommended: No (exploratory engineering study); the Hydra configs + this blueprint serve as the pre-commitment.
