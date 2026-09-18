# Phase 2 — Search layer: LoRA rank / scaling / learning rate / meta-learned LoRA precedents
(Agent report, 2026-09-16. Numbers marked "verified from extracted PDF text" by the agent for PERK and rsLoRA; others from fetched HTML. Every reference below goes through Phase 2 source verification before use in Phase 3.)

## 1. microsoft/LoRA reference implementation (https://github.com/microsoft/LoRA, loralib/layers.py)
- result += (lora_dropout(x) @ lora_A.T @ lora_B.T) * scaling, scaling = lora_alpha / r.
- Init: A kaiming_uniform (a=sqrt(5)), B zeros (Linear/MergedLinear). Embedding class reversed.
- Paper (arXiv:2106.09685) §4.1: Gaussian A, zero B, scale α/r; "we simply set α to the first r we try and do not tune it".
- GPT-3 175B Table 5: adapt {Wq,Wv} r=8 best; all four attention mats r=2 about as good; MLP never adapted. Table 6: r=1..64 essentially flat for SFT.
- LRs: RoBERTa-base 4e-4–5e-4 (α=8, r=8); RoBERTa-large 2e-4–4e-4; GPT-2 M/L 2e-4 (α=32, r=4); GPT-3 175B LoRA lr 2e-4 vs full FT ~5e-6 (~40×).

## 2. sayakpaul gist "Make a high-rank LoRA low-rank" (https://gist.github.com/sayakpaul/9bae12402eddd53a79ee1f64b659b07b)
- Post-hoc rank-reduction of a trained PEFT-format LoRA (keys *.lora_A.weight [r,in], *.lora_B.weight [out,r]); not a training tool.
- svd_low_rank_lora.py: randomized_svd(matrix, rank, niter=5); reduce_lora_rank(lora_A, lora_B, niter, new_rank=4) → ΔW=B@A, SVD, B_new=U√S, A_new=√S·Vh; reduce_lora_rank_state_dict(...); compare_approximation_error(...) prints relative Frobenius error per module; fire CLI main(repo_id, filename, new_rank, niter=None, check_error=False, new_lora_path=None). Hardcodes .to("cuda").
- low_rank_lora.py: random (JL) projection variant — lossy, prefer SVD.
- Use in pipeline: train slow LoRA at generous rank (64–128), SVD-truncate to k∈{4,8,16,32}, evaluate meta-test loss per k without retraining → effective-rank curve; also inspect singular spectrum of B@A per layer. Caveats: truncated high-rank ≠ optimum at low rank; gist ignores `scaling`; A≠0,B≠0 after truncation.

## 3. "LoRA Without Regret" (Thinking Machines, Sept 2025, https://thinkingmachines.ai/blog/lora/)
- W' = W + (α/r)BA, α=32, A uniform scale 1/√d_in, B=0, same LR for A and B.
- Optimal LoRA LR ≈ 10× full-FT LR (fit multiplier 9.8; ~15× for ~100-step runs).
- 1/r scaling makes optimal LR ≈ rank-independent; identical early learning curves regardless of rank; exception rank=1.
- "Attention-only LoRA significantly underperforms MLP-only LoRA, and does not further improve performance on top of LoRA-on-MLP." Apply to all layers, esp. MLP.
- Capacity: low ranks track full FT then plateau when dataset exceeds adapter capacity (~2 bits/param). LoRA less tolerant of large batch sizes than full FT (independent of rank).

## 4. rsLoRA (arXiv:2312.03732), LoRA+ (arXiv:2402.12354), DoRA (arXiv:2402.09353)
- rsLoRA Thm 3.2: scale must be Θ(1/√r) or learning collapses/unstable for large r; replace α/r with α/√r. Llama-2-7B, AdamW 5e-5, ranks {4..2048}, all linear modules; α/r "collapsing gradients with higher ranks". Gain is capacity, not LR.
- LoRA+: η_B/η_A = 16 default with standard init; "fix λ=16 and tune only η_A"; r=α=8 (GLUE), r=64 α=16 (Llama-7B); 1–2% gains, up to 2× faster.
- DoRA: r=32 α=64 on q,k,v,up,down; lr 2e-4 vs LoRA 3e-4. Adds a magnitude parameter group; not obviously useful here.

## 5. "LoRA Learns Less and Forgets Less" (arXiv:2405.09673)
- Llama-2-7B; r∈{16,64,256}; α=2r; Attention vs All modules; CPT 0.25B–20B tokens. Best LRs: code IFT full 5e-5 / LoRA r16,64 2e-4 / r256 1e-4; math IFT full 1e-5 / LoRA 1e-4 / r256 5e-5; math CPT full 1e-5 / LoRA 4e-5.
- "LoRA's best learning rates should be set one order of magnitude higher than that of full finetuning, often ranging between 5e-5 and 5e-4."
- LoRA substantially underperforms full FT for CPT even at r=256/20B tokens; full-FT ΔW rank 10–100× typical LoRA ranks, grows with data; MLP ΔW higher rank than attention; use All modules, r=256 for CPT-scale data.

## 6. Prior work: LoRA as the meta-learned / slow parameter
- PERK (ICLR 2026, arXiv:2507.06415): outer loop meta-learns LoRA init; base frozen; inner loop = CLM on context chunks *in the LoRA* (4 steps differentiable AdamW via Higher, per-layer-per-step learned inner LRs init 5e-5, truncated unrolling last 1–2 steps). GPT-2-127M, Qwen2.5-0.5B (7B/8B top 4 layers). r=256 all modules, rsLoRA, α→256, dropout 0.1. Outer AdamW lr 1e-5, wd 0.01, cosine, 3% warmup. Train 8K, extrapolate to 64K–128K. Closest precedent (LoRA slow, LoRA fast).
- EMG MAML+LoRA (arXiv:2601.04181): meta-learned shared LoRA init r=4, frozen backbone, K=4 inner steps.
- MASS (arXiv:2603.03524): inner 2-step LoRA SFT, outer meta-learns scorer/generator.
- MetaLoRA/MetaPEFT (CVPR 2025, github.com/doem97/metalora), AutoLoRA (arXiv:2403.09113): bi-level meta-learning of LoRA hyperparameters, not slow-weight LoRA.
- Reverse configuration (slow full, fast LoRA): FocuSFT (arXiv:2605.09932): fast LoRA r=32/α=64 on gate/up/down of top 35% layers, zero-reinit each step, K=2 inner steps at inner lr 1.0 with clip 1.0; slow = full Qwen2.5-7B AdamW 1e-5. Provable Meta-Learning with LoRA (arXiv:2410.22264): meta-learns base W so rank-r adapters adapt well. TTT-NTP (arXiv:2606.21803): MLP down-projection as rank-1-write fast weight with slow d×d projection. SEAL (arXiv:2506.10943): inner LoRA r=128 α=16, outer RL.
- No paper found with exactly: LoRA slow via meta-gradient + big full-rank MLP fast.

## Agent's recommendation (input to Phase 3, not final)
- Rank sweep r∈{16,64,256}, start 64; include 256 because outer objective is CPT-scale; SVD truncation (gist) to read effective rank.
- Scaling: rsLoRA α/√r with α=16 (≈2 at r=64), fixed α across sweep; or α=2r with standard α/r. Standard init (A kaiming, B=0) so outer loop starts exactly at the pretrained/TTT-naive point.
- Outer LR: literature ratio LoRA≈10× full-FT; TTT-E2E full-weight anchors 3e-3 (125M pretrain), 4e-4 (ext); PERK/FocuSFT meta-LoRA used 1e-5. Bracket: 125M sweep {3e-4, 1e-3, 3e-3, 1e-2}; 760M {1e-4, 3e-4, 1e-3, 3e-3}. Keep β=(0.9,0.95), wd 0.1, clip 1.0, 10% warmup, cosine→1e-5. Optionally LoRA+ (η_B=4–16×η_A). Prefer smaller outer batch at higher LR (LoRA batch-size sensitivity).
- Attention LoRA: ablation only; MLP-only first, then MLP+{Wq,Wv} at half rank per matrix at equal param budget.
- Note (orchestrator correction): the agent wrote "125M model with d_ff=3072"; the e2e 125M config uses intermediate_size 2048 (1664 in the prime-MLP variant).
