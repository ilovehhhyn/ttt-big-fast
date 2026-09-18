# Phase 2 — Search layer: inner-loop optimizer for large fast weights under differentiate-through meta-learning
(Agent report, 2026-09-16. LaCT and e2e claims read from cloned code; Muon from Keller Jordan's post/repo and Moonshot arXiv:2502.16982; others from arXiv HTML. All references go through Phase 2 verification.)

## 1. LaCT — "Test-Time Training Done Right" (arXiv:2505.23884; code github.com/a1600012888/LaCT)
- Inner rule (Eq. 8): weight-update(W,g) = L2-Normalize(W − g); with Muon (Eq. 9): L2-Normalize(W − Muon(g)), g = Σ_i η_i ∇ℓ_i over the chunk; optional per-token predicted momentum scalar. Code default for LLM: use_muon=False, use_momentum=True (Muon is the ablation). Update: dw = bmm(...); optional momentum; if use_muon: zeropower_via_newtonschulz5(dw); w += dw; w = w/(‖w‖_row+1e-5)·w_norm ("channel-wise l2 norm, conceptually like post-norm"). NS: bf16, 5 steps, per-step coefficients, Frobenius-normalized input; comment "adding detach here sometimes improves stability".
- Chunk sizes: 2048 default for LM (2K–4K in LM experiments), up to 1M tokens for novel-view synthesis.
- Fast weights: SwiGLU (W1,W2,W3), "12d² per block, totaling 40% of model weights as fast weights"; "state-to-parameter size ratio ≥40%, an order of magnitude larger than previous methods' 0.1% to 5%".
- LR: per-token per-matrix lr = softplus(Linear(x)+inv_softplus(base_lr)), base_lr=0.001; multiplies each token's gradient before summation (token importance).
- Normalization rationale (verbatim): "Fast-weight updates in TTT repeatedly accumulate gradients, and thus suffer from magnitude explosion or decayed memory." "...our fast-weight normalization is analogous to the post-layer norm in Transformer architectures". "Muon normalizes the spectral norm of matrix gradient using Newton-Schulz iterations"; "the learning rate now only reflects the relative importance of tokens within a chunk as Muon normalizes the absolute scale"; "Muon also improves the numerical stability in our setup."
- Differentiate-through: plain torch ops under torch.compile; autograd double-backprops through NS and renorm.
- Ablation Fig. 7(b): "Muon's surprising effectiveness over Vanilla Gradient Descent and Momentum" (no numbers in text); note "Vanilla GD" still includes the L2 weight renorm.

## 2. Muon (kellerjordan.github.io/posts/muon/; github.com/KellerJordan/Muon; arXiv:2502.16982; tridao.me/blog/2026/gram-newton-schulz/)
- M ← βM + g (β=0.95, Nesterov), O = NS5(u), W ← W − lr·O·scale; Keller: scale = max(1, rows/cols)^0.5, lr 0.02 "in units of spectral norm per update". Moonshot: W_t = W_{t−1} − η(0.2·O·√max(A,B) + λW); Lemma 1: orthogonalized update RMS = 1/√max(A,B), so 0.2·√max(A,B) matches AdamW update RMS 0.2–0.4 → reuse AdamW LR/wd.
- NS: X = G/‖G‖_F (transpose if tall); 5 × {A = XXᵀ; B = bA + cA²; X = aX + BX} with (3.4445, −4.7750, 2.0315), bf16 stable; 10 steps no better.
- Cost: ≈ 20a²b + 10a³ FLOPs per NS5 (a=min dim); Keller's overhead formula T·m/B → 0.5–0.7% in pretraining (B≈0.5–16M tokens/step) but ≈750% at B=1024 tokens/step (TTT). Example: 1536×4096 fast weight: NS5 ≈ 230 GFLOP vs 12.9 GFLOP chunk forward (≈6× the chunk's fwd+bwd through that matrix). LaCT amortizes with 2K–1M chunks and square d×d weights.
- Differentiability: 15 matmuls, autograd fine (LaCT does it). Agent analysis (no paper found on double-backward through NS): polar-factor Jacobian has 1/(σ_i+σ_j) terms; NS5 polynomial slope at 0 ≈ 485–493, so backward amplifies small-singular-value directions up to ~500× in bf16 — most ill-conditioned of the candidates; Frobenius pre-normalization needs eps.

## 3. Titans (arXiv:2501.00663) / Atlas (arXiv:2505.23735) / Nested Learning (arXiv:2512.24695)
- Titans: S_t = η_t S_{t−1} − θ_t ∇ℓ(M_{t−1};x_t); M_t = (1−α_t)M_{t−1} + S_t; KVB loss; data-dependent η,θ,α; MLP memory ≥2 layers; ℓ2-norm on keys/queries only; no gradient normalization; momentum adds one mn state.
- Atlas: Omega rule over sliding window of c losses; M_t = α_t M_{t−1} − η_t NewtonSchulz-k(S_t); S_t = θ_t S_{t−1} + ∇. Table 6 ablation: Atlas 19.97 ppl / 52.77 acc; "w/o Muon" 19.65 ppl / 52.56 acc → Muon not a clear win; polynomial features (22.14 w/o) and window c>1 (21.98 at c=1) matter more.
- Nested Learning / HOPE: optimizers as nested associative memories; Continuum Memory System; no specific inner-normalization detail beyond adaptive η_t.

## 4. Adam inside a differentiated inner loop
- higher (github.com/facebookresearch/higher, higher/optim.py): "Some of the adaptative gradient-style differentiable optimizers may be unstable and yield NaNs when taking higher order gradients"; DifferentiableAdam masks exp_avg_sq==0 via backward hook (d√v/dv → ∞ at v=0); DifferentiableRMSprop "suffers from gradient correctness issues".
- Agent derivation: at inner step 1 with bias correction Δ = g/(|g|+ε) ≈ sign(g); dΔ/dg = ε/(|g|+ε)²: ≈0 for |g|≫ε, ≈1/ε=1e8 for |g|≪ε → meta-gradient through step 1 dead or exploding per element; with 8–128 steps and β₂=0.999, v̂ stays sample-dominated. Raising ε to 1e-4…1e-2 turns Adam into soft per-element-clipped SGD.
- Metz et al. 2019 (arXiv:1810.10180): gradients through unrolled optimization "either strongly biased (short truncations) or exploding norm (long truncations)".
- Andrychowicz et al. 2016 (arXiv:1606.04474): drop second derivatives; preprocess gradients as (log|g|/p, sign g).
- MAML++ (arXiv:1810.09502): plain SGD inner loop + learned per-layer per-step LRs (LSLR) + multi-step loss. Meta-SGD (arXiv:1707.09835): learned per-parameter LR vector = stateless learned diagonal preconditioner. iMAML (arXiv:1909.04630): implicit differentiation, needs near-converged inner problem (not our regime).
- No paper found benchmarking SGD vs Adam vs normalized SGD inside a MAML-style unrolled loop at scale; all practical differentiate-through TTT systems (TTT-Linear/MLP, TTT-E2E, LaCT, Titans, Atlas) use SGD-family updates.

## 5. Normalized SGD / clipping in TTT
- TTT-E2E: paper says "gradient descent" + mini-batching for stability; repo: optax.chain(clip_by_global_norm(1.0), sgd(lr=1, momentum=None)); global norm over ALL prime-MLP weights → ‖ΔW_all‖_F = min(‖g‖,1) per chunk; for a 1536×3328 matrix at init std 0.02 (‖W‖_F≈45) caps relative change ≈2% per chunk. ilr warmup 0.1→1 optional. Full backprop through all T/b steps with checkpointing through time.
- TTT-Linear/MLP (arXiv:2407.04620): W_t = W_{t−1} − η(x)∇ℓ, η(x)=η_base·σ(θ_lr·x), η_base 1.0 (Linear)/0.1 (MLP), b=16, no clipping; stability from LN + residual inside inner model; Adam left to future work.
- LaCT: normalizes the weights (row L2 renorm) not the gradient, plus optional Muon.
- E²-TTT (arXiv:2608.21308; github.com/zeyun-zhong/E2-TTT): GD with learned lr (0.01) and learned weight decay (0.1), optional momentum; no Adam/Muon.

## 6. Per-step cost, m×n fast weight (a=min, b=max, chunk B tokens)
| Optimizer | Persistent state | Saved per step for outer backward | Extra compute |
|---|---|---|---|
| SGD + global clip | 0 | 1 scalar | O(mn) |
| Normalized SGD per matrix | 0 | 1 scalar | O(mn) |
| AdamW | 2mn | 2mn + sqrt/div graph | ~10mn elementwise; double-backward through 1/(√v+ε) |
| Muon (+momentum) | mn (0 if orthogonalizing raw gradient) | NS intermediates 5×(2a²+ab) or recompute | 20a²b+10a³ FLOPs (e.g. 230 GFLOP for 1536×4096 vs 12.9 GFLOP chunk fwd at B=1024) |
Baseline: inner gradient ≈ 6·B·mn FLOPs/step; fast weights (mn) must already be stored/recomputed per chunk boundary. AdamW triples per-step footprint; Muon adds 1.5–2× plus NS FLOPs.

## Agent's ranked recommendation (input to Phase 3)
1. Clipped/normalized SGD, no state (TTT-E2E recipe), optionally per-matrix: W ← W − η·g/max(1, ‖g‖_F/τ), τ=1, η=1; per-matrix variant W ← W − η_c·g/(‖g‖_F+1e-6) with optional LaCT-style learned η_c = η_base·softplus(w·x̄_c+b), η_base ≈ 1–5% of ‖W‖_F per chunk; optional LaCT row-wise L2 renorm of W instead of weight decay if drift over 128 steps is a problem. Backward (I−ĝĝᵀ)/‖g‖ well conditioned when ‖g‖≫ε.
2. Normalized SGD + learned diagonal/per-row LR (Meta-SGD/MAML++ LSLR): stateless Adam-like per-coordinate scaling, differentiable, no sqrt/eps; adds outer params.
3. Muon (LaCT Eq. 9) with Moonshot RMS scaling: best inner quality in LaCT at 2K–1M chunks; at B=1024 ~6× the chunk's own fwd+bwd, NS intermediates stored/recomputed, backward ill-conditioned (~500× in bf16); Atlas ablation shows no ppl gain. Worth it only at ≥8K chunks or square d×d fast weights.
4. AdamW: ruled out — 2mn state checkpointed per step; first-step derivative ε/(|g|+ε)² dead/exploding at ε=1e-8; needs higher-style hooks; raising ε≈1e-3 makes it a 3×-memory worse-conditioned normalized SGD.
