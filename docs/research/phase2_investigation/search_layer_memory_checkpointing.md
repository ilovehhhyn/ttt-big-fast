# Phase 2 — Search layer: inner-loop memory & checkpointing for large fast weights
(Agent report, 2026-09-16; code of e2e, ttt-lm-jax, PyTorch-MAML, LaCT read locally. Notation: T seq len, b chunk, N=T/b, |W| bytes of one fast-weight copy, |v| inner-optimizer state, A_step activations of one chunk step.)

## 1. RW-TTT (arXiv:2605.28053) — serving only
Batched inference server for request-owned fast weights (Qwen3-4B with In-Place TTT checkpoint, Feng et al. ICLR 2026 arXiv:2604.06169; fast weights = MLP down-projection). "In-place" = per-request FastWeightState in an owner-indexed table mutated at chunk boundaries (chunk 128 tokens); "checkpointing" = rollback snapshot before speculative writes, not gradient checkpointing. PyTorch eager + Triton. First-order, no outer loop. Peak 33.53 GiB with 8 streams. Not relevant to training memory.

## 2. MixFlow-MG (arXiv:2505.00793, ICML 2025; NOT 2505.17895 = DataRater, which merely uses MixFlow-MG)
Reparameterizes inner update Υ(∇L, θ, v, η, x) and uses Hessian symmetry to turn vector-Hessian products into HVPs computed forward-over-reverse via custom_vjp (`fwdrev_grad`); plus block remat and `checkpoint_name(d_params,'inner_grads')`. Their memory formula with per-step checkpointing: O(|A| + T·(|θ|+|v|)) — it attacks |A| (second-order activations), not T·(|θ|+|v|). Reported ~4× less memory (80% of configs), >10× peak, ≤25% wall-clock; T∈{2,4,6,8} inner Adam steps, 44M–16B. JAX (custom_vjp + jvp), PyTorch snippet in appendix, no repo. Composes with scan+remat in principle but needs re-derivation per inner optimizer; with N=128 and large W the T·|W| term dominates → little benefit. Not recommended.

## 3. PyTorch activation checkpointing blog / memory-budget API
Plain AC saves region inputs; SAC adds policy_fn; memory budget API (`torch._dynamo.config.activation_memory_budget`, compile-only, experimental) lets the AOTAutograd min-cut partitioner pick a policy over one compiled region. No notion of loops over time (unrolled loop traced flat). **Unusable for the meta-gradient region: torch.compile + create_graph=True unsupported** (pytorch/pytorch#91469 open since Dec 2022; current builds raise "aot_autograd does not currently support double backward"; maintainer statement Apr 2025 on discuss.pytorch.org). Usable only for the frozen prefix (first-order).

## 4. Axolotl activation_offloading
Requires gradient_checkpointing; modes true (TRL offloader, CPU with stream overlap) / legacy / disk / hidden_states (saved-tensor hook offloading per-decoder-layer checkpoint input). Tied to HF layer structure, first-order fine-tuning. The reusable primitive is `torch.autograd.graph.saved_tensors_hooks` / `save_on_cpu`, not Axolotl.

## 5. PyTorch-MAML (shirleyzhu233; models/maml.py L158–185)
Checkpoints through time, one inner step per region: `state = cp.checkpoint(_inner_iter_cp, episode, *state)`, state = params + momentum. Uses reentrant checkpoint → first-pass/detach hack (`is_first_pass`, create_graph only on recompute). README: activations O(N)→O(1), "up to 80% GPU memory with ~20% more time"; parameter copies remain O(N·(|θ|+|v|)). PyTorch equivalent of e2e's scan+remat with group size 1; no √N grouping, no offload. Today use `use_reentrant=False` (docs 2.14: "supports all ways of performing the backward pass", backward inside region allowed) so the hack is unnecessary.

## 6a. TTT-Linear/MLP (arXiv:2407.04620; ttt-lm-jax ttt_layer.py)
Mini-batch b=16; dual form avoids materializing per-token W; Appendix C: "we still need to save T/b W_s at the end of the mini-batches ... gradient checkpointing ... we apply it through time." Code: `jax.lax.scan(jax.remat(partial(jax.lax.scan, f), prevent_cse=False), carry, x_grouped)` with `remat_mini_batch_group_size` — identical two-level structure to e2e's `scan_remat_chunk`. Inner gradient analytic → outer AD first-order over an explicit formula (e2e instead does true grad-of-grad).

## 6b. LaCT (arXiv:2505.23884; github.com/a1600012888/LaCT)
Chunks 2048 (760M LM) / 4096 (3B) at 32K; 1M for NVS. Fast weights = full SwiGLU MLP, learnable init, no LoRA, up to 40% of params (LM configs 0.75d²). Inner: W ← L2-Normalize(W − Muon(g)), NS5 ("30·b·d³ FLOPs"), optional momentum, per-token learned lrs. **Memory management: none beyond chunk size** — plain Python for-loop `w1 = w1 + dw1`, no checkpoint, inner gradient hand-derived (`silu_backprop`) so outer backward is first-order autograd retaining every W_i as a saved bmm input → O(N·|W|), tolerable because N=16 (32K/2K). Fast weights fp32 master + bf16 compute; Nov-2025 fused Triton kernels. torch.compile only on small helpers.

## 7. Large-fast/small-slow designs and per-chunk-copy mitigations
- In-Place TTT (arXiv:2604.06169, ICLR 2026; github.com/ByteDance-Seed/In-Place-TTT): fast = W_down of MLP every 6th layer, chunk 512–1024, one first-order GD step, inner-product loss; W_i = W_0 + Σ_{j<i}Δ_j via parallel prefix scan; no meta-gradient; 0.5B–14B, 32K–128K.
- TTT-NTP (arXiv:2606.21803; github.com/yancyou/TTT-NTP): fast at MLP down-proj in 6/32 layers, chunk 1024, rank-one accumulated writes, first-order.
- FocuSFT (arXiv:2605.09932; github.com/JarvisPei/FocuSFT): inverse design (fast LoRA r=32 on FFN top 35% layers, K=2, lr 1.0, clip 1.0, first-order; slow full 7B); 1.71× wall time.
- MASS (arXiv:2603.03524v2): LoRA inner loop on 8B, 2 steps, second-order, uses MixFlow-MG + block remat; no code.
- GradMem (arXiv:2603.13875; github.com/yurakuratov/gradmem): second-order through ≤5 steps, tiny fast state; needed custom double-backward because attention kernels lack it.
- WAM-TTT (arXiv:2607.06988): learned fast-init per layer (slow), small fast MLP, N=1 inner step, second-order; DeepSpeed ZeRO-2.
- TTT-E2E: "increase gradient checkpointing through time by a factor of log(T)"; hidden state 88M vs 18M at 760M.
- Reversible inner updates (Maclaurin, Duvenaud & Adams 2015, arXiv:1502.03492): exact reversal of SGD-with-momentum storing lost bits; not applicable to clipped momentum-free SGD (circular / non-invertible); no 2024–26 TTT work uses it.
- bf16 W / host offload: no TTT paper reports either; LaCT and e2e keep fp32 fast weights (`state_dtype: fp32`). bf16 with lr=1 updates over 128 steps is a numerics risk (no stochastic rounding).
- Nobody found meta-learns a LoRA-only slow model with >50% params fast. Closest: LaCT (40% fast, everything slow), In-Place TTT / TTT-NTP (large fast, first-order).

## What e2e costs and the levers (from jax_utils.py::scan_remat_chunk, config.py)
- With inner_remat_freq=n: outer scan saves the carry (model in state_dtype, inner opt state, eqx state) at N/n boundaries; backward of one group recomputes n carries + one step's residuals. **Peak ≈ (N/n + n)·(|W|+|v|) + A_step per sequence × vmapped local batch.** Default n=1 → N copies. E.g. 500M fast params fp32 = 2 GB: 256 GB/seq at 128K; n=√N≈11 → ~23 copies ≈ 46 GB; treeverse (JAX docs O(log₂D) memory for O(log₂D)× FLOPs; `eqx.internal.scan(kind="checkpointed")` online treeverse with checkpoints=O(√max_steps) default) → ~7–8 copies ≈ 15 GB.
- Backward also carries an adjoint of size |W| plus HVP temporaries: budget ≥3–4 live |W| buffers.
- Adam/AdamW inner adds |v|=2|W| to every copy; Muon without momentum is stateless (LaCT's choice).
- Host offload natively: `jax.checkpoint(policy=jax.checkpoint_policies.save_and_offload_only_these_names(names_which_can_be_saved=[], names_which_can_be_offloaded=[...], offload_src="device", offload_dst="pinned_host"))` with `jax.ad_checkpoint.checkpoint_name` (docs.jax.dev/en/latest/201/memory-spaces.html; "memory-kind support varies by platform"; parameter offloading "works only when scanning over axis 0"). e2e already imports jax.ad_checkpoint and dispatches named policies.

## Ranked recommendation — JAX / e2e
1. Keep scan-through-time remat; set inner_remat_freq ≈ √N (11 at 128K/1K; 6 at 32K). Zero new code; N → ~2√N copies. Keep inner SGD or momentum-free Muon (|v|=0); W fp32.
2. Offload group-boundary W snapshots to pinned host: wrap fast-weight carry leaves with `checkpoint_name(..., "fast_w")`, replace the outer group's nothing_saveable policy with `save_and_offload_only_these_names([], ["fast_w"], "device", "pinned_host")`. Device then holds ~n copies; host N/n. Verify with `Compiled.memory_analysis()`; confirm platform support. Ensure slow LoRA params are closed over, not carried.
3. Nested remat / treeverse (third scan level or `eqx.internal.scan(kind="checkpointed")`) → O(log N) copies at O(log N)× recompute; only if 2 unavailable.
4. state_dtype=bf16 for carried W: halves copies but changes inner dynamics; last resort.
5. Do not pursue MixFlow-MG or reversible updates.

## Ranked recommendation — PyTorch reimplementation
1. Mirror e2e in eager: Python loop over chunks; each group of n steps in `torch.utils.checkpoint(step_group, W, x, use_reentrant=False)`; inner gradient via `torch.func.grad` / `autograd.grad(create_graph=True)` over `torch.func.functional_call`; memory (N/n+n)·|W| + A_step; nest one more level for log depth if needed.
2. Offload boundary W_i with `torch.autograd.graph.save_on_cpu(pin_memory=True)` around the outer loop.
3. No torch.compile in the meta-gradient region (#91469); compile only the frozen prefix; fused attention lacks double backward (GradMem needed a custom one) → SDPA math path or restrict fast weights to MLPs.
4. If the fast module is a fixed SwiGLU MLP, LaCT's analytic-inner-gradient pattern makes the outer backward first-order so compile/Triton become usable, at the cost of hand-derived gradients.
5. Fast weights fp32 with bf16 matmul casts; momentum-free inner optimizer; Muon NS5 = 30·d³ FLOPs per matrix per chunk that the outer backward must traverse.
