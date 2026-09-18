# Big-Fast / Small-Slow TTT-E2E — Proposal and Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether TTT-E2E still works when the fast (inner-loop) weights are the full MLP matrices of a pretrained Transformer and the slow (outer-loop) weights are a small meta-learned set (LoRA on attention, norm gains, learned inner learning rates), evaluated with the paper's DCLM (8K) / PG-19 (32K) protocol.

**Architecture:** Fork `test-time-training/e2e` (JAX/Equinox). Load Llama-3.2-1B weights into the e2e Transformer (adding GQA and Llama-3 RoPE scaling), run it with sliding-window attention (k=8192), and reuse the existing chunked inner loop (`scan_remat_chunk`, `SWA` KV-cache, `Evaluator`). New code: a `LoRALinear` module, parameter specs that make base MLPs inner-loop and LoRA/norms/inner-LRs outer-loop, two new inner optimizers (strictly normalized SGD, differentiable AdamW), a decay-toward-W0 knob, a forgetting probe, and Hydra configs for six experimental arms.

**Tech Stack:** JAX 0.5.x + Equinox + Optax + Grain + Hydra + W&B (as in e2e); `uv`; Princeton Della (della-pli H100 80 GB, della-gh GH200 96 GB); HF `safetensors` for weight import; DCLM parquet and PG-19 from Hugging Face, tokenized to zarr locally (free).

**Spec:** `docs/research/FINDINGS.md` (sections 1–10) plus `docs/research/phase1_scoping/research_question_brief.md` (H1, arms, user directives). This plan copies every decided value verbatim; anything not decided is listed in §0.4 "Open items".

---

## 0. Proposal (decisions, exact values, and what is still unknown)

### 0.1 Hypothesis and arms
**H1**: On a pretrained sliding-window Transformer, a small slow set (attention LoRA + RMSNorm gains + per-tensor inner learning rates) meta-learned through the inner loop captures most of the gain that TTT-E2E's all-weights outer loop provides over TTT-naive, while the fast weights are the model's own MLP matrices (initialized from pretraining, reset every sequence). **Falsifier**: arm C ≈ arm B while arm D ≫ arm B.

| Arm | Fast (inner) weights | Slow (outer) weights | Role |
|---|---|---|---|
| A. SWA baseline | none | none (frozen Llama-3.2-1B, SWA k=8192) | floor |
| B. TTT-naive | MLPs (w1,w2,w3) of the fast blocks | none | dynamic evaluation; isolates meta-learning |
| C. **Proposed** | same as B | LoRA on wq,wk,wv,wo of **all** blocks + all RMSNorm gains + per-tensor inner-LR scalars; MLP LoRA (w1,w2,w3) only in ablation C-mlp | main arm |
| D. Full-slow | same as B | all parameters (paper-style) | upper bound on what LoRA gives up |
| E. Paper TTT-E2E | `feed_forward_prime`, last 1/4 | all parameters | reference; released `1b_ttt_e2e_*` checkpoints |
| F. Small-fast control | paper-style prime MLP (last 1/4), cloned from the pretrained MLP | same slow set as C | C-vs-F isolates fast-weight size from regime |

Fast-weight fraction sweep (arms B, C, D): last 1/4 (4 blocks, 201M params) → last 1/2 (8 blocks, 403M) → all (16 blocks, 805M). Start at 1/4.

### 0.2 Fixed values (copied from FINDINGS.md and the user's decisions)
| Item | Value |
|---|---|
| Base model | `meta-llama/Llama-3.2-1B`: 16 layers, hidden 2048, intermediate 8192, 32 heads, 8 KV heads (GQA), head_dim 64, vocab 128256, RoPE θ=500000 with Llama-3 scaling (factor 32, low_freq_factor 1, high_freq_factor 4, original_max_position 8192), RMSNorm eps 1e-5, tied embeddings, **no** QK-norm, **no** post-norm, SwiGLU MLP. Gated repo; licence permits research. |
| Tokenizer / data | Llama-3 tokenizer. **Free path (decided):** DCLM-Baseline from HF `mlfoundations/dclm-baseline-1.0-parquet`, keep docs with ≥8193 Llama-3 tokens, tokenize, write zarr `/train` and `/val` (Task 2). 32K stage on **PG-19** (`deepmind/pg19`, public domain) instead of Books3. Paper's GCS buckets not used (Requester-Pays). Arm E's released checkpoint is scored on *our* val split; its published numbers are not compared directly. |
| Sequence mixer | SWA k=8192 everywhere (`seq_modeling_block: SWA`); at 8K identical to full attention. |
| Chunk (inner mini-batch) b | 1024 tokens. |
| Contexts | Stage 1 meta-training + eval at 8K on DCLM; Stage 2 extension + eval at 32K on PG-19 (mandatory). |
| Run order | **A → C → E → B → D → F** (A is evaluation-only and gives the reference number; C needs the Task 9 sweep first). No mechanical dependency between arms. |
| Reset | Fast weights reset to W0 at every sequence boundary (train and eval), as in the paper. |
| Inner loss / outer loss | Chunk i loss with W_{i−1}, then step; outer loss = mean over chunks of loss-before-update. |
| Inner optimizer arm N (required) | Per-tensor normalized SGD: `W ← W − η_rms·√(numel)·g/(‖g‖_F + 1e-6)`, with a floor: if ‖g‖_F < 1e-6 skip the step. η_rms sweep {3e-4, 1e-3, 3e-3} (paper-equivalent per-element RMS ≈ 1/√11.5M ≈ 3e-4 is the 1× point). Global-norm variant as reference. Learned per-tensor multiplier `exp(θ_t)`, θ_t init 0, in the slow set. Inner-LR warmup 0.1→1.0 over the first 10% of outer steps. |
| Inner optimizer arm A (required) | AdamW ε=1e-8 differentiated through: m,v carried in fp32; warm-start m0=g1, v0=g1² from the first chunk; denominator `sqrt(v̂ + ε²)` (finite gradient at v=0); β1=0.9, β2=0.9 (8–32 steps), weight decay 0; lr sweep {3e-4, 1e-3, 3e-3} (Adam update RMS ≈ lr, same unit as η_rms). Kill criterion: NaN in any step, or outer-grad-norm > 10× the arm-N run at the same step for 20 consecutive steps. |
| Inner optimizer arm M (optional) | Muon without momentum: NS5 coefficients (3.4445, −4.7750, 2.0315), 5 iterations, bf16, scale 0.2·√max(m,n), same η sweep. 125M-token budget only. |
| Decay toward W0 | ΔW ← (1−λ)·ΔW before each inner step, λ ∈ {0, 0.05, 0.2}. |
| Forgetting probe | After TTT on a sequence, evaluate W_T and W0 on one fresh held-out 8K DCLM chunk; report mean ΔNLL (W_T − W0). Plus the paper's per-token-index loss curve. |
| LoRA | rank r ∈ {16, 64, 256}, start 64; rsLoRA scaling α/√r, α=16; A ~ Uniform(±1/√fan_in), B = 0; on wq, wk, wv, wo of all 16 blocks (attention LoRA at r=64 ≈ 13.6M params); C-mlp ablation adds w1,w2,w3. LoRA merged into weights for deployment accounting. |
| Outer optimizer | AdamW β=(0.9, 0.95), wd 0.1 on LoRA A/B only (0 on norm gains and inner-LR scalars), clip 1.0, 10% linear warmup, cosine to 1e-5. LR sweep {3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2}. |
| Batch size | **0.5M tokens per outer step** at both stages (64 seqs × 8K; 16 seqs × 32K) — the paper's pretraining batch, not its doubled 1M extension batch, because LoRA is less batch-tolerant. Per-device vmapped batch = 1 (`accum_steps` = per-device sequences). |
| Token budgets | Hyperparameter sweeps: 125M tokens (250 steps). Headline runs: 1.3B tokens (2600 steps) = 5% of the 1.3B model's Chinchilla budget, mirroring the paper's fine-tuning recipe. |
| Memory | `state_dtype fp32`; `inner_remat_freq` = 3 at 8K, 6 at 32K; per-arm expected fp32 carry per sequence at 32K: 1/4 → 9 GB (N) / 27 GB (A); all → 36 GB / 109 GB (needs `n_state_parallel=2..4` or pinned-host offload). |
| Seeds | 3 seeds for arms C and D at the chosen hyperparameters; 1 seed elsewhere. Paper's noise floor 0.001 nats. |
| Cost accounting | H100-hours per arm incl. inherited pretraining; deployed (post-merge) params and inference FLOPs/token; training sec/1K tokens. |

Compute estimate (planning only): second-order TTT ≈ 3.4× a standard step; 1.24B params → ≈2.5e10 FLOPs/token → 1.3B-token headline run ≈ 3.3e19 FLOPs ≈ 57 H100-hours at 160 TFLOP/s effective; a 250-step sweep run ≈ 6 H100-hours. Full grid (3 η × 2 optimizers × 3 ranks × 6 LRs is too many): sweep sequentially — η first (6 runs), then LR at best η (6 runs), then rank (2 more), then λ (2 more) ≈ 16 sweep runs ≈ 100 H100-hours, plus 6 arms × 3 fractions headline ≈ 1000 H100-hours upper bound; prune fractions 1/2 and all to arm C only if budget is tight.

### 0.3 Notes on tricky areas (from discussion)
- Normalized SGD must divide by the norm every step; the paper's clip-to-1 leaves small-gradient steps unnormalized.
- LoRA on the fast MLP is only a rank-r shift of W0 — hence attention/norms/inner-LR as the slow set and MLP-LoRA as an ablation.
- Keep W0 in fp32; a unit-norm step over 10⁸ elements moves each by ~1e-4, at bf16 resolution for 0.02-scale weights.
- With all 16 blocks fast the prefix shortcut disappears (no frozen prefix computed once with flash attention); expect ≳3× the paper's training latency.
- Flash/cuDNN attention has no double backward: the fast (suffix) blocks must use XLA attention (`force_flash: False`).
- The KV cache in the suffix blocks carries activations computed with older fast weights; this matches the paper and is not "fixed".
- "Matched parameters" = deployed post-merge params + inference FLOPs/token; LoRA params reported separately.

### 0.4 Open items (must be resolved before the corresponding task; none block Task 1)
1. **Della**: partition names, GPUs/job, walltime, scratch quota (`sinfo`, `sacctmgr show assoc user=hh9077`); Duo login prevents remote probing.
2. **Pinned-host offload**: whether the JAX/CUDA on Della supports `save_and_offload_only_these_names(... "pinned_host")`; only needed for all-blocks AdamW.
3. **Arm E checkpoint**: `gs://ttt-e2e-checkpoints/1b_ttt_e2e_pretrain_dclm_8k_1x_cc` is Requester-Pays (a few GB, ≈$1); alternative is emailing the authors. Data buckets are no longer needed.
4. **HF gated access** to `meta-llama/Llama-3.2-1B` (accept licence, `HF_TOKEN`).
5. **Arm E at 32K**: the authors released `1b_ttt_e2e_pretrain_dclm_8k_1x_cc` and `1b_ttt_e2e_finetune_books_8k_1x_cc`, not a 1B 32K checkpoint; arm E at 32K = rerun `ext-1b-e2e-32K` from the released pretrain checkpoint (1250 steps × 1M tokens in their recipe; use the same for comparability).
6. **Arm E vs A/C parameter matching**: arm E is the paper's 1B architecture (24 layers, d 2048, ff 4352+prime), not Llama-3.2-1B; it is a reference row, not a matched comparison. Decide whether to also train a Llama-3.2-shaped E (expensive: from-scratch meta-pretraining).

---

## Global Constraints
- Python ≥3.12, `jax[cuda12]<0.6`, `equinox>=0.11.12`, `optax>=0.2.4`, `grain>=0.2.7`, `orbax-checkpoint>=0.11.13`, `zarr>=3.0.4` (from e2e `pyproject.toml`); add `safetensors>=0.4`, `huggingface_hub>=0.25`, `transformers>=4.45` (dev only, for parity test), `pytest`.
- `compute_dtype bf16`, `param_dtype fp32`, `state_dtype fp32`.
- All new config knobs have defaults that reproduce e2e behaviour exactly when unset.
- Every experiment is a Hydra `+experiment=` file under `configs/experiment/llama1b/…`; W&B project `ttt-big-fast`.
- Commit after every task; never commit data, checkpoints, or `HF_TOKEN`.

---

## File Structure

Fork of `e2e` at repo root (`ttt/`, `configs/`), plus:
- `ttt/model/lora.py` — `LoRALinear` (wraps `NormalLinear`), init, rsLoRA scaling, merge.
- `ttt/model/attention.py` — add GQA (`num_kv_heads`) and Llama-3 RoPE scaling.
- `ttt/model/transformer.py` — config-driven norm/QK-norm flags already exist; add `fast_blocks` handling for `suffix_len == num_hidden_layers` (empty prefix), LoRA insertion, inner-LR scalars, ΔW decay.
- `ttt/optimizers.py` — `normalized_sgd`, `differentiable_adamw`, `muon_nomom` inner optimizers.
- `ttt/infra/hf_import.py` — Llama-3.2 safetensors → e2e pytree.
- `ttt/eval/forgetting.py` — forgetting probe.
- `ttt/config.py` — new fields.
- `configs/model/llama1b.yaml`, `configs/training/llama1b/{meta-8K,ext-32K}.yaml`, `configs/experiment/llama1b/{A_swa,B_naive,C_lora,C_mlp,D_full,F_smallfast}-{8K,32K}.yaml`.
- `tests/` — parity, LoRA, optimizer, memory tests.
- `scripts/della/` — sbatch templates; `scripts/memory_probe.py`.

---

### Task 1: Environment on Della and e2e smoke test

**Files:**
- Create: `scripts/della/env.sh`, `scripts/della/smoke.sbatch`
- Modify: `pyproject.toml` (add safetensors, huggingface_hub, transformers, pytest)

**Interfaces:** Produces a working `uv run --exact train +deploy=interactive training.dummy_dataset=true …` on one GPU.

- [ ] **Step 1: Fork and add deps**
```bash
git clone https://github.com/test-time-training/e2e.git ttt-big-fast-src && cd ttt-big-fast-src
git remote rename origin upstream
```
Edit `pyproject.toml` dependencies to add `"safetensors>=0.4", "huggingface_hub>=0.25", "transformers>=4.45"`.
- [ ] **Step 2: Write `scripts/della/env.sh`**
```bash
#!/bin/bash
module purge; module load cudatoolkit/12.8 cudnn/cuda-12.x/9.8.0 2>/dev/null || true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export JAX_COMPILATION_CACHE_DIR=/scratch/gpfs/$USER/jax_cache
export HF_HOME=/scratch/gpfs/$USER/hf
uv sync --exact
```
(Module names are a placeholder for the exact Della module list — confirm with `module avail cudatoolkit cudnn` on login; this is Open item 1.)
- [ ] **Step 3: Write `scripts/della/smoke.sbatch`** running the dummy-dataset e2e pretrain config for 20 steps on 1 GPU:
```bash
#!/bin/bash
#SBATCH -J ttt-smoke -N1 --gres=gpu:1 -c8 --mem=64G -t 00:30:00
source scripts/della/env.sh
uv run --exact train +deploy=interactive +experiment=125m/pretrain/pretrain-125m-e2e \
  training.dummy_dataset=true training.total_steps=20 training.log_wandb=false backend.num_devices=1
```
- [ ] **Step 4: Run it; expected: 20 steps complete, loss printed, no NaN.** Record sec/step.
- [ ] **Step 5: Commit** `chore: fork e2e, della env + smoke test`.

---

### Task 2: Data on Della (free path: HF parquet → zarr)

**Files:** Create `scripts/prep_dclm.py`, `scripts/prep_pg19.py`, `tests/test_prep.py`, `configs/deploy/della.yaml`

**Interfaces:** Produces zarr stores `/scratch/gpfs/$USER/data/dclm8k/{train,val}` and `.../pg19/{train,val}` in the exact layout `ttt/dataloader/lm_dataset.py::Dataset` reads: one 1-D int32 array per split, documents concatenated, each document = `[128000 (BOS)] + tokens`, no padding.

- [ ] **Step 1: Failing test**
```python
def test_zarr_layout(tmp_path):
    write_split(tmp_path/"val", docs=[[5,6,7],[8,9]], bos=128000)
    a = zarr.open_array(zarr.storage.LocalStore(str(tmp_path)), path="/val")
    assert a.dtype == np.int32 and a[:].tolist() == [128000,5,6,7,128000,8,9]
def test_filter_keeps_only_long_docs():
    assert keep_doc(n_tokens=8193, min_tokens=8193) and not keep_doc(8192, 8193)
```
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement `scripts/prep_dclm.py`**: `datasets.load_dataset("mlfoundations/dclm-baseline-1.0-parquet", streaming=True, split="train")`; tokenizer `AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")`; for each `text`: tokenize without special tokens, keep if `len ≥ 8193`; assign to val if `hash(doc_id) % 200 == 0` (≈0.5%), else train; stop when train has `target_tokens` (default 2.0e9, enough for 1.3B headline + sweeps) and val has ≥ 64×8193×4 tokens; write with `zarr` BloscCodec zstd clevel 3 (same codec as e2e). Multiprocess tokenization (`num_proc = SLURM_CPUS_PER_TASK`). Log docs seen / kept.
- [ ] **Step 4: Implement `scripts/prep_pg19.py`**: same, from `deepmind/pg19` (`train`/`validation` splits already defined by the dataset), keep books with ≥ 32769 tokens, target 1.5e9 train tokens.
- [ ] **Step 5: `configs/deploy/della.yaml`**: `deploy_paths.data.dclm_filter_8k: /scratch/gpfs/${oc.env:USER}/data/dclm8k`, `deploy_paths.data.books3: /scratch/gpfs/${oc.env:USER}/data/pg19` (key name kept so upstream configs still resolve), `checkpoint: /scratch/gpfs/${oc.env:USER}/ckpt`.
- [ ] **Step 6: Run** both scripts as a CPU sbatch job (`-c 32 --mem 128G -t 12:00:00`; expect several hours: long docs are a few percent of DCLM, so the stream has to pass roughly 100B tokens of parquet to collect 2B). Verify `zarr.open_array(...)[:5]` starts with 128000.
- [ ] **Step 7: Commit** `feat: free data pipeline (DCLM parquet + PG-19 → zarr)`.

---

### Task 3: Llama-3.2-1B in the e2e Transformer (GQA, RoPE scaling, import, parity)

**Files:**
- Modify: `ttt/config.py` (`num_kv_heads: int | None = None`, `rope_scaling: str = "none"`, `rope_scale_factor: float = 32.0`, `rope_low_freq_factor: float = 1.0`, `rope_high_freq_factor: float = 4.0`, `rope_original_max_pos: int = 8192`)
- Modify: `ttt/model/attention.py` (`AttentionBase.__init__`, `_split_heads`, `core_attention_op`, `SWA.__call__`, `precompute_freqs_cis`)
- Create: `ttt/infra/hf_import.py`, `configs/model/llama1b.yaml`, `tests/test_llama_parity.py`

**Interfaces:** Produces `load_llama_into_metamodel(model: MetaModel, repo_id: str) -> MetaModel` and config `llama1b`.

- [ ] **Step 1: Write the failing parity test**
```python
# tests/test_llama_parity.py
import jax, jax.numpy as jnp, numpy as np, torch, pytest
from transformers import AutoModelForCausalLM
from ttt.infra.hf_import import build_llama_metamodel
@pytest.mark.slow
def test_logits_match_hf():
    ids = np.array([[128000, 791, 6864, 315, 9822, 374, 12366, 13]], dtype=np.int32)
    hf = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", torch_dtype=torch.float32)
    ref = hf(torch.tensor(ids)).logits[0].detach().numpy()
    model, state = build_llama_metamodel("meta-llama/Llama-3.2-1B", seq_len=8192)
    out = model.language_model(state, batch_from_ids(ids[0]))  # helper in test file
    got = np.asarray(out.logits, dtype=np.float32)
    assert np.abs(got - ref).max() < 0.5 and np.corrcoef(got.ravel(), ref.ravel())[0,1] > 0.999
```
- [ ] **Step 2: Run:** `pytest tests/test_llama_parity.py -m slow -x` → FAIL (ImportError).
- [ ] **Step 3: Implement GQA** in `AttentionBase`: `self.num_kv_heads = config.num_kv_heads or config.num_attention_heads`; `wk`, `wv` shaped `(hidden, num_kv_heads*head_dim)`; `_split_heads_kv`; in `core_attention_op` and `SWA.__call__` repeat K/V heads `jnp.repeat(xk, num_heads//num_kv_heads, axis=1)` before `dot_product_attention` (SWA cache shape becomes `(window, num_kv_heads*head_dim)`; update `init_kv_cache`).
- [ ] **Step 4: Implement Llama-3 RoPE scaling** in `precompute_freqs_cis`: for each inverse frequency `f`, wavelength `λ=2π/f`; `low=orig/low_freq_factor`, `high=orig/high_freq_factor`; if `λ>low`: `f/=factor`; elif `λ<high`: unchanged; else smooth: `s=(orig/λ − low_ff)/(high_ff−low_ff)`, `f = (1−s)·f/factor + s·f`. Gate on `config.rope_scaling == "llama3"`.
- [ ] **Step 5: Write `hf_import.py`**: download safetensors via `huggingface_hub.snapshot_download`; map `model.embed_tokens.weight→wte`, `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight.T→blocks.seq_modeling_block.w{q,k,v,o}.weight[i]`, `mlp.gate_proj→w1`, `mlp.up_proj→w3`, `mlp.down_proj→w2` (all transposed to `(in,out)`), `input_layernorm→seq_norm`, `post_attention_layernorm→ffn_norm`, `model.norm→ln_f`; stack per-layer arrays along axis 0 to match the vmapped `Block` pytree; use `eqx.tree_at`.
- [ ] **Step 6: Write `configs/model/llama1b.yaml`**: `num_hidden_layers 16, hidden_size 2048, intermediate_size 8192, num_attention_heads 32, num_kv_heads 8, vocab_size 128256, bos 128000, eos 128001, rms_norm_eps 1e-5, tie_word_embeddings True, qk_norm False, post_norm False, rope_theta 500000, rope_scaling llama3, prime False`.
- [ ] **Step 7: Run the parity test → PASS.** Also run with `seq_modeling_block: SWA, sliding_window_size 8192` on an 8K sequence and assert identical logits to `self_attention` (tolerance 1e-3).
- [ ] **Step 8: Commit** `feat: llama-3.2-1b import with GQA and llama3 rope scaling`.

---

### Task 4: LoRA module and slow/fast parameter specs

**Files:** Create `ttt/model/lora.py`, `tests/test_lora.py`; Modify `ttt/config.py` (`lora_rank: int = 0`, `lora_alpha: float = 16.0`, `lora_scaling: str = "rslora"`, `lora_targets: list[str] = ["wq","wk","wv","wo"]`), `ttt/model/attention.py`/`transformer.py` (wrap targeted `NormalLinear`s), `ttt/utils/filter_utils.py` (no change expected; verify specs match).

**Interfaces:** `LoRALinear(base: NormalLinear, rank, alpha, scaling, key)` with `.lora_A`, `.lora_B`, `__call__(x)`, `merged() -> NormalLinear`. Spec strings: inner `["language_model.**.suffix_blocks.feed_forward.w1.weight", "...w2.weight", "...w3.weight"]`; outer `["language_model.**.lora_A", "language_model.**.lora_B", "language_model.**_norm.weight", "language_model.model.ln_f.weight", "inner_lr_log"]`.

- [ ] **Step 1: Failing tests**
```python
def test_lora_zero_init_is_identity():
    base = NormalLinear(cfg, 8, 4, std=0.02, key=k1); l = LoRALinear(base, rank=2, alpha=16.0, scaling="rslora", key=k2)
    x = jax.random.normal(k3, (3, 8)); assert jnp.allclose(l(x), base(x))
def test_rslora_scale():
    assert LoRALinear(base, rank=64, alpha=16.0, scaling="rslora", key=k2).scale == pytest.approx(16/8)
def test_merge_matches_forward():
    l = eqx.tree_at(lambda m: m.lora_B, l, jax.random.normal(k4, l.lora_B.shape))
    assert jnp.allclose(l.merged()(x), l(x), atol=1e-5)
def test_specs_partition_llama():
    model, _ = build_llama_metamodel(..., lora_rank=4, fast_blocks=4)
    inner = model.inner_parameters(); outer = model.trainable_parameters()
    assert count(inner) == 4*3*2048*8192 and count(outer) < 20_000_000 and no_overlap(inner, outer)
```
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement `LoRALinear`**: fields `base`, `lora_A (rank,in)` init `Uniform(±1/√in)`, `lora_B (out,rank)` zeros, `scale = alpha/√rank` (rslora) or `alpha/rank`; `__call__ = base(x) + (x @ lora_A.T @ lora_B.T) * scale` in `compute_dtype`; `merged()` returns `NormalLinear` with `weight + scale * (lora_B @ lora_A).T`.
- [ ] **Step 4: Insert** in `AttentionBase.__init__` when `config.lora_rank>0` for names in `lora_targets`; in `SwiGLUMLP.__init__` for `w1,w2,w3` if listed. Add `inner_lr_log: jnp.ndarray` (one scalar per fast tensor, zeros) to `MetaModel`. Add `fast_blocks: int` config → sets `suffix_len`; handle `suffix_len == num_hidden_layers` by making `prefix_blocks=None` and skipping `prefix_call` (feed `xt_embed` directly to the suffix scan).
- [ ] **Step 5: Run → PASS. Commit** `feat: LoRA slow adapters, fast-MLP specs, inner-lr scalars`.

---

### Task 5: Inner optimizers (normalized SGD, differentiable AdamW, Muon-no-momentum) and ΔW decay

**Files:** Modify `ttt/optimizers.py`, `ttt/config.py` (`OptimizerType` add `normalized_sgd`, `dadamw`, `muon_nomom`; fields `norm_scope: "tensor"|"global"`, `lr_rms`, `eps: 1e-8`, `b1 0.9`, `b2 0.9`, `warm_start: True`, `delta_decay: 0.0`); `ttt/model/transformer.py::inner_loop_step` (apply decay, per-tensor learned LR, warm-start on chunk 0). Create `tests/test_inner_optimizers.py`.

**Interfaces:** `make_optimizer(cfg, ilr_multiplier, inner_lr_log=None) -> optax.GradientTransformation` with the same `init/update` contract e2e uses.

- [ ] **Step 1: Failing tests**
```python
def test_normalized_sgd_step_norm():
    tx = make_normalized_sgd(lr_rms=1e-3, scope="tensor", eps=1e-6)
    g = {"a": jnp.ones((4,8))*7.0, "b": jnp.ones((2,2))*1e-9}
    u, _ = tx.update(g, tx.init(g), g)
    assert jnp.allclose(jnp.sqrt(jnp.mean(u["a"]**2)), 1e-3, rtol=1e-4)   # per-element RMS == lr_rms
    assert jnp.all(u["b"] == 0)                                             # floor: skip tiny gradients
def test_dadamw_first_step_finite_metagrad():
    tx = make_dadamw(lr=1e-3, b1=0.9, b2=0.9, eps=1e-8, warm_start=True)
    def f(g): u, _ = tx.update({"w": g}, tx.init({"w": g}, first_grad={"w": g}), {"w": g}); return jnp.sum(u["w"]**2)
    g = jnp.array([0.0, 1e-12, 1.0]); assert jnp.all(jnp.isfinite(jax.grad(f)(g)))
def test_muon_orthogonal():
    o = newton_schulz5(jax.random.normal(k, (64, 128)).astype(jnp.bfloat16)); s = jnp.linalg.svd(o.astype(jnp.float32), compute_uv=False)
    assert jnp.abs(s - 1).max() < 0.3
def test_delta_decay_moves_toward_w0(): ...  # W0=1, W=2, λ=0.5, zero grad → W=1.5
```
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement**
```python
def make_normalized_sgd(lr_rms, scope, eps=1e-6, lr_mult=None):
    def update(g, state, params=None):
        if scope == "global":
            n = optax.global_norm(g); tot = sum(x.size for x in jax.tree.leaves(g))
            u = jax.tree.map(lambda x: jnp.where(n < eps, 0.0, -lr_rms*jnp.sqrt(tot)*x/(n+eps)), g)
        else:
            def one(x, m): n = jnp.sqrt(jnp.sum(x*x)); return jnp.where(n < eps, 0.0, -lr_rms*m*jnp.sqrt(x.size)*x/(n+eps))
            u = jax.tree.map(one, g, lr_mult or jax.tree.map(lambda _: 1.0, g))
        return u, state
    return optax.GradientTransformation(lambda p: optax.EmptyState(), update)
```
`make_dadamw`: state `(m, v, count)`; `init(params, first_grad=None)` sets `m=first_grad, v=first_grad**2, count=1` when `warm_start`; `update`: `m=b1*m+(1-b1)*g; v=b2*v+(1-b2)*g**2; mh=m/(1-b1**t); vh=v/(1-b2**t); u=-lr*mh/jnp.sqrt(vh+eps**2)`. `newton_schulz5(G)`: normalize by `‖G‖_F+1e-7`, transpose if rows>cols, 5× `A=X@X.T; B=b*A+c*A@A; X=a*X+B@X`, transpose back; `make_muon_nomom`: `u = -lr_rms*0.2*jnp.sqrt(max(m,n))*newton_schulz5(g)` — hmm: scale so per-element RMS≈lr_rms: `u = -lr_rms*jnp.sqrt(max(m,n))*ns(g)` (NS output RMS = 1/√max(m,n)).
- [ ] **Step 4: Wire into `inner_loop_step`**: before the gradient, if `delta_decay>0`: `W = W0 + (1−λ)(W − W0)` (W0 = the outer model's fast params, available as `model` in `process_suffix_chunk`); pass `lr_mult = jax.tree.map(jnp.exp, inner_lr_log)` for arm N; on chunk 0 with `dadamw.warm_start`, init state from the first gradient.
- [ ] **Step 5: Run → PASS. Commit** `feat: normalized SGD, differentiable AdamW, Muon-nomom inner optimizers; ΔW decay`.

---

### Task 6: Forgetting probe and per-token curves in eval

**Files:** Create `ttt/eval/forgetting.py`, `tests/test_forgetting.py`; Modify `ttt/model/loop.py::Evaluator` (add `forgetting` loader over a second held-out shard; `MetricType.forgetting_delta_nll`).

**Interfaces:** `forgetting_delta_nll(model, state, seq, probe_seq) -> float`: runs `loss_for_sequence` on `seq` to obtain W_T (return the final carry), then computes mean NLL of `probe_seq` (first 8K tokens, no TTT, chunked forward with W_T) minus mean NLL with W0.

- [ ] **Step 1: Failing test** — with `lr_rms=0` the delta is exactly 0; with `lr_rms=1e-2` on a dummy dataset the delta is finite and logged.
- [ ] **Step 2: Implement**: extend `loss_for_sequence` with `return_final_model: bool`; probe uses `train_mode="pretrain"`-style chunk scan with `state_prefix,state_suffix` reset.
- [ ] **Step 3: Log** `forgetting_delta_nll` and the existing `token_nll_loss` curve to W&B at every eval. Commit.

---

### Task 7: Experiment configs for all arms

**Files:** `configs/training/llama1b/meta-8K.yaml`, `configs/training/llama1b/ext-32K.yaml`, `configs/experiment/llama1b/*.yaml` (A_swa, B_naive, C_lora, C_mlp, D_full, F_smallfast × {8K, 32K}), `configs/experiment/llama1b/E_paper_1b_32K.yaml` (= upstream `ext-1b-e2e-32K` with `resume_exp_name` pointing at the released checkpoint).

Values (verbatim from §0.2): `seq_length 8192 / 32768`, `global_batch_size 64 / 16`, `total_steps 250 (sweep) | 2600 (headline)`, `optimizer_outer: adamw, lr ${outer_lr}, b1 0.9, b2 0.95, weight_decay 0.1, clip 1.0, lr_warmup_steps = 10% of total, lr_decay_steps = total, end_lr 1e-5`, `optimizer_inner: {normalized_sgd | dadamw | muon_nomom}, lr_rms ${eta}, eps 1e-8`, `ilr_init 0.1, ilr_warmup_steps = 10% of total`, `model: llama1b, seq_modeling_block SWA, sliding_window_size 8192, mini_batch_size 1024, force_flash False, fast_blocks {4|8|16}, lora_rank {0|16|64|256}, lora_alpha 16, lora_targets [wq,wk,wv,wo] (+[w1,w2,w3] for C_mlp)`, `training.inner_remat_freq 3 / 6`, `accum_steps = per-device seqs`, `spec_inner / spec_outer` per arm (A: `train_mode pretrain`, no inner; B: inner only, `spec_outer: ["nothing"]` → outer LR 0; D: `spec_outer ["**"]`; F: `prime True, suffix_len 4, intermediate_size 8192`, prime MLP initialized by cloning the pretrained MLP in `hf_import.py`).

- [ ] **Step 1:** Write the 13 YAML files. **Step 2: Test:** `python -c` that composes every experiment with Hydra (`--cfg job`) without error and asserts `spec_inner`/`spec_outer` match ≥1 parameter each (the repo already asserts this). **Step 3: Commit.**

---

### Task 8: Memory validation before any sweep

**Files:** `scripts/memory_probe.py`, `tests/test_memory_probe.py`

- [ ] **Step 1:** Script that builds arm C at a given `fast_blocks`, `seq_length`, `inner_remat_freq`, optimizer, compiles `train_on_sequence` with `jax.jit(...).lower(...).compile()` on dummy data and prints `compiled.memory_analysis()` (temp + argument + output bytes) and then runs 3 real steps recording `jax.local_devices()[0].memory_stats()["peak_bytes_in_use"]`.
- [ ] **Step 2:** Run the grid {4, 8, 16 blocks} × {8K, 32K} × {normalized_sgd, dadamw} on one H100; write results to `docs/research/memory_measured.md` next to the §0.2 estimates. Any config >75 GB: set `n_state_parallel` 2 or 4 (Task 8b) or enable offload (Open item 2).
- [ ] **Step 3: Commit.**

---

### Task 9: Hyperparameter sweeps (125M tokens each, 8K, arm C, fast_blocks=4)

Order (each stage fixes the previous winner by DCLM val loss):
1. η_rms ∈ {3e-4, 1e-3, 3e-3} × {normalized_sgd, dadamw} (6 runs) at outer LR 1e-3, r=64.
2. Outer LR ∈ {3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2} (6 runs).
3. Rank ∈ {16, 256} (2 runs; 64 already done).
4. λ ∈ {0.05, 0.2} (2 runs; 0 already done).
5. Optional: Muon-nomom at best η (1 run); C-mlp ablation (1 run).
Also arm B's own inner sweep (3 runs) so it is not a straw man.
- [ ] Sweep launcher `scripts/della/sweep.sbatch` (array job, one config per index, W&B tags `sweep/<stage>`); results table `docs/research/sweep_results.md`. Commit after each stage.

---

### Task 10: Headline runs and evaluation

- [ ] Run in the order **A, C, E, B, D, F**. 8K stage: arms A, C, B, D, F at fast_blocks=4, 1.3B tokens, 3 seeds for C and D; arm E = released 1B checkpoint scored with `eval_mode=true` on our val split (Open item 3). Eval: DCLM val loss, per-token curve, forgetting ΔNLL, peak memory, sec/1K tokens.
- [ ] 32K stage: initialize from the 8K checkpoints (`load_part=params`), PG-19, 16×32K batch, 2600 steps; same order A, C, E, B, D, F. Same metrics on PG-19 val at 32K.
- [ ] Fast-fraction ablation: arm C (and B) at fast_blocks 8 and 16 at 32K.
- [ ] SVD effective-rank curve on the r=256 checkpoint (port of the sayakpaul gist to NumPy: truncate ΔW=scale·B@A to k∈{4,8,16,32,64}, evaluate val loss per k).
- [ ] Results: `docs/research/results.md` with the arm table (loss, Δ vs A, forgetting, deployed params, FLOPs/token, H100-hours) and the per-token plots. Commit.

---

## Self-review
- Spec coverage: H1/arms (Tasks 7, 9, 10); SWA + chunk 1024 (Task 3/7); reset per sequence (unchanged e2e); both optimizer arms with exact formulas (Task 5); LoRA placement/rank/scaling/LR (Tasks 4, 7, 9); batch size and budgets (§0.2, Task 7); forgetting probe + λ (Tasks 5, 6); 32K mandatory (Task 10); memory plan and measurement (§0.2, Task 8); fast-fraction ablation (Task 10); cost accounting (Task 10); model choice and data (Tasks 2, 3). Arm E's parameter mismatch and the 1B-32K checkpoint gap are Open items 5–6, not silently resolved.
- Placeholders: Della module names in Task 1 are explicitly an open item; no other TBDs.
- Type consistency: `make_normalized_sgd(lr_rms, scope, eps, lr_mult)`, `make_dadamw(lr, b1, b2, eps, warm_start)`, `newton_schulz5`, `LoRALinear(base, rank, alpha, scaling, key)`, `build_llama_metamodel(repo_id, seq_len, lora_rank, fast_blocks)`, `forgetting_delta_nll(model, state, seq, probe_seq)` used consistently.
