# Plan: hypotheses, arms, protocol, and what comes next

This file states what the project tests, how each test is decided, what is running, and what
is planned, with the predictions written before the results. Numbers with their intervals are
in `results.md`; this file names decisions. Updated 2026-09-23.

## Hypothesis

H1. On a pretrained sliding-window Transformer whose fast weights are the full MLPs of the last
4 blocks, a small slow set (LoRA r = 64 on `wq, wk, wv, wo, w1, w2, w3`, the RMSNorm gains, one
learned log step size per fast tensor) meta-learned through the inner loop captures most of the
gain that test-time training can give.

The measurable form is a 2x2 (`scripts/two_by_two.py`). Slow weights trained {through the inner
loop, by plain fine-tuning with the inner loop off} are each evaluated {with TTT on, with TTT
off} on the same sequences. The quantity H1 is about is the interaction: does training through
the inner loop make TTT MORE useful? A training effect that is the same with TTT off is a better
set of slow weights, not a better use of fast weights. Every effect is paired per sequence and
clustered by document (`ttt/eval/paired.py`).

## Arms

| arm | fast weights | slow weights | status |
|---|---|---|---|
| A | none (inner rule `none`) | none | measured at 8K, 32K, 128K |
| B | MLPs of the last 4 blocks | none | measured; inner-rate sweeps for normalized SGD, AdamW, Muon, preconditioned SGD |
| C | same | LoRA + norms + learned step sizes | measured at 10 and 40 steps (window 8192 and 1024, PG-19 and SlimPajama); 725-step runs queued |
| D | same | every parameter (`--arm D`, slow spec `**`) | 10-step feasibility only; needs its own outer-rate sweep at `truncate_bptt=1` |
| E | the released TTT-E2E 760M checkpoint | evaluation only | scored once at 8K; not at 32K |
| F | a small extra "prime" MLP per fast block (paper layout; width `--prime-intermediate`, LaCT output RMSNorm and zero gate), its W_0 meta-learned | as C plus the gate | implemented 2026-09-23 (code only); never run on Llama |

## Protocol (what every 32K number shares unless a table says otherwise)

| item | value |
|---|---|
| model | `unsloth/Llama-3.2-1B` (a mirror of Llama-3.2-1B), 16 blocks, fp32 master weights, bf16 forward |
| context, chunk | T = 32768, b = 1024; window k = 8192 (reference) or 1024 (round 2); k >= b always |
| inner rule | normalized SGD, per-tensor, `lr_rms` = per-element RMS of the update; unit 1/sqrt(n_fast) = 7.05e-5 |
| inner rate | 4e-6 for every loss-comparison table (the loss optimum); larger rates in the memory tables |
| outer rule | AdamW, lr 4e-4, warmup 10%, cosine to 1e-5, weight decay 0.1 on LoRA only, clip 1.0 |
| sequences per step | 4 (131,072 tokens); the reference uses 32 (1,048,576) |
| truncation of the meta-gradient | 2 chunks at window 8192, 4 at window 1024 (memory bound, see `operations.md`) |
| evaluation | 32 validation sequences from 22 PG-19 books, fixed shuffle seed 0; 96 sequences from 88 documents on SlimPajama |
| loss | nats per token, mean over chunks of the loss BEFORE each update (TTT-E2E Eq. 6) |
| reference budget | 725 steps x 32 sequences = 760,217,600 tokens (their 32K extension) |

## Decision rules used so far

| question | measurement | rule |
|---|---|---|
| is a gain memory or repair of the window? | `scripts/context_value.py`: what old context is worth to a HEALTHY model (full attention, restart every S tokens) | a TTT gain above that ceiling is repair, not memory |
| does the model remember text it cannot see? | `scripts/recall_probe.py`: plant a passage, repeat it beyond attention's reach L(k-1); recall = loss drop on the repeat | without TTT recall must be exactly 0 (checked on every run); full attention is the ceiling |
| does a new write rule help? | recall and ordinary loss at several inner rates | wrong if recall is not at least doubled at a loss no worse than normalized SGD's |
| does a 2-GPU run equal a 1-GPU run? | `scripts/compare_runs.py --metrics loss --tol 1e-3` | the orchestrator refuses to submit the long jobs otherwise |
| is a matched-budget corpus large enough? | `train.json` token count against 760,217,600 | refuse rather than let the loader cycle |

## What the results decided (short)

1. Window 8192 on PG-19 leaves almost nothing for a memory to gain (+0.0208 nats). Round 2
   moved to window 1024 (+0.0992), SlimPajama (arXiv +0.2548 at 1024) and 128K.
2. The interaction is positive everywhere and small (+0.0015 at 8192; +0.0126 then +0.0083 at
   1024; +0.0074 on SlimPajama). Across SlimPajama documents it follows what old context is
   worth, not the window damage.
3. Memory measured directly is 4% of full attention with the original write; the write is the
   limit (recall is flat along the passage; later updates erase little).
4. Muon at 1.2e-4 gives 38% of full attention at almost unchanged loss, and 56% at 2.4e-4
   for +0.116 in loss. The preconditioned rule (shared key directions removed) gives 26% and
   stays cheap. Both confirm the diagnosis.
5. At the reference window the prize stays small however long the LoRA trains: what TTT adds
   on the same weights falls from +0.0248 (8 steps) to +0.0067 (150 steps). The interaction
   there grew from +0.0015 (10 steps, 17/22) to +0.0058 (20 steps, 22/22) and +0.0077 (60
   steps, 21/22); the 150-step 2x2 will say whether it keeps growing (prediction: +0.005 to
   +0.010).

## Queued on Della (all resumable; each chain link resumes from the checkpoint)

The four matched-budget chains moved on 2026-09-23 from the `arora` queue (estimated start
about a week out, A100s) to the PLI partition (`--account=pli_x --partition=pli --qos=pli-low`,
H100s, estimated start 2026-09-24 evening, no preemption). Same settings and result paths;
the `arora` copies (14237614 to 14237617, 14237645 to 14237648) were cancelled first.

| jobs | run | resources | purpose |
|---|---|---|---|
| 14330761, 14330762 | `C32k_match`: arm C, window 8192, 725 x 32, normalized SGD 4e-6, `--eval-ttt-off` | 4 H100s, 48 h per link | the reference budget |
| 14330763, 14330764 | `C32k_match_ctl`: same with `--inner none` | 4 H100s, 30 h | its plain fine-tuning control |
| 14330765, 14330766 | `Ck1024_match`: window 1024, `truncate_bptt=4`, same budget | 4 H100s, 24 h | the reference budget where the ceiling is larger |
| 14330767, 14330768 | `Ck1024_match_ctl` | 4 H100s, 16 h | its control |
| 14169729 | `C32k_bs32s60`: 60 steps of 32 sequences (63M tokens), running since 2026-09-23 09:55 | 1 GPU, 22 h | closest single-GPU approach to the reference regime |
| 14330259 | `C32k_ctl150` (4 GPUs): `--inner none` control for the 150-step run (`C32k_ctl60` finished: 2.5255) | `gpu-test`, 1 h | the 2x2 at 150 steps |
| 14333213 to 14333217 | `C_32k_k1024_muon_s40`: 40 steps of arm C through Muon at 1.2e-4, window 1024, truncation 4, `--eval-ttt-off`; chain of five 1-hour links | `gpu-test` | THE next experiment; its 2x2 partner is the existing `C_32k_k1024_ctl_s40` |
| login GPU | `scripts/della/login_cells_k8192_s60_s150.sh`: plain 60- and 150-step weights with TTT on and off (waits for `C32k_ctl150`) | 2 x 13 min | the 2x2 at 60 and 150 steps, window 8192 |

All of these use the ORIGINAL write rule (normalized SGD at 4e-6). They test the thesis at the
reference budget with the weak memory. Keep them; add a Muon run at that budget once Muon
meta-training is validated and its orthogonalization is sped up.

## Read on 2026-09-23 (details in `results.md` and R, "2026-09-23")

- The window-8192 ladder finished: 20 steps at 4/8/16/32 sequences, 60 and 150 steps. What
  TTT adds on the same weights shrinks with training (+0.0125 at 60 steps, +0.0067 at 150),
  under the +0.0208 ceiling; 150 steps of 4 sequences (2.4664) beat 20 steps of 32 (2.5664) at
  equal tokens; truncation 1 equals truncation 2 (-0.0004 per book).
- Muon at 2.4e-4 on the 40-step weights: recall +1.5005 (56% of full attention), loss 2.7937.
- Weights trained at normalized SGD 2e-5: recall +0.5201, loss 2.7243 (trained at 4e-6 and
  tested at 2e-5: +0.3880, 2.8318).
- The first Muon meta-training validations (jobs 14247911, 14247912, then 14330212) never
  trained: `--steps 3` and `--steps 5` both make the 10% warmup round to zero (Python rounds
  0.5 down), a hard error by design whose message had named the wrong fix. Message fixed;
  resubmitted with `--steps 6`.

Read a job with `grep -E '^\[eval\]|^ttt_on|step' /scratch/gpfs/ARORA/hh9077/logs/<name>-<job>.out`.

## Next experiment (the most important one)

Meta-train the LoRA through Muon. It separates the two explanations for the small interaction:
either H1 is wrong, or the write was too weak for the slow weights to have anything to shape.

1. Done on 2026-09-23 17:45: the validation fit (286 s per step, peak 60.4 GiB at truncation
   4), and the 40-step run `C_32k_k1024_muon_s40` was submitted as a chain of five 1-hour
   `gpu-test` links (jobs 14333213 to 14333217). Its control is the existing
   `C_32k_k1024_ctl_s40` (same outer settings; a plain fine-tune does not depend on the inner
   rule). Expected to finish the same night.
2. Score both sets of weights with Muon at 1.2e-4: recall (`scripts/recall_probe.py`,
   `--gap 17408 --pairs 32`) and the 2x2 (`scripts/two_by_two.py`).
3. Prediction, written 2026-09-23 before the run: loss at or below 2.6777 with TTT on; recall
   at or above +1.0241; interaction at least twice +0.0083. If recall stays near +1.0241 and
   the interaction near +0.0083, meta-learning adds nothing to a strong fixed write rule, and
   the thesis is not supported at this scale.

## Planned after that, in order (Helen's order, decided 2026-09-23 from a reading of LaCT)

Measurement order once Della is reachable (all on the 40-step window-1024 weights unless
said otherwise): (a) bf16 Newton-Schulz speed and recall against fp32; (b) `row_reset` recall and
loss at Muon 1.2e-4 and 2.4e-4; (c) memory probe at chunk 2048, window 2048; (d) 6-step
validations of `--token-rates` and of arm F, then their 40-step pairs.

LaCT is "Test-Time Training Done Right" (arXiv 2505.23884). A sub-agent read the full PDF on
2026-09-23 and compared it with this project row by row; Helen chose these five items.

1. A chunk of 2048 tokens (LaCT App. C.2: 2048 for its 760M model, 4096 for 3B). The window
   must stay at or above the chunk, so this arm runs at window 2048 or larger, where old
   context is worth +0.0667 on PG-19 (not +0.0992 at window 1024). It is also the principled
   fix for Muon's cost: five Newton-Schulz iterations cost about 30 x hd x state FLOPs (LaCT
   App. A, Eq. 17 and 18), so Muon is cheaper than the token computation only when the chunk
   exceeds (5/3) hd tokens, about 3400 for our 2048 x 8192 matrices. At chunk 1024 that is
   the measured 20 s per sequence against 3.4. Needs a memory probe and its own plain control.
2. Arm F (a small extra fast MLP, the paper's layout) built with LaCT's RMSNorm on the fast
   output followed by a zero-initialised gate (LaCT Alg. 2, App. C.3), so the new memory
   leaves the pretrained model untouched at step 0. Built 2026-09-23 (`--arm F
   --prime-intermediate 2048`; the prime W_0 is trained by the outer loop through
   `TrainConfig.fast_init_trained`). Not yet run: needs a memory probe, a 6-step validation
   and a 40-step pair at window 1024.
3. Per-token learning rates as a slow parameter: eta_t = softplus(Linear(x_t) + bias), one
   small linear layer per fast block, meta-learned through the inner loop (LaCT Eq. 4,
   Alg. 1 and 2). Under a normalized or Muon rule they only weight tokens against each other
   inside a chunk (LaCT Sec. 3.2). This is the cheaper form of "the LoRA weights the fast
   update" (the meta-learned preconditioner below). Built 2026-09-23 (`--token-rates`; eta = 1 at init, so a
   fresh run equals one without the flag). Not yet run.
4. L2 row normalization of the fast weights after each inner step, no weight decay (LaCT
   Alg. 1 and 3, Sec. 3.2: each row of W - g is rescaled to the row norm of W). Test first at
   evaluation time on the existing 40-step weights with the recall test. The hope is a larger
   stable inner rate, which caps every memory result so far. Built 2026-09-23
   (`--weight-norm row_reset`). Not yet run.
5. The bf16 Newton-Schulz iteration, built 2026-09-23 (`--ns-dtype bfloat16`; default
   float32 unchanged). On a 2048 x 8192 matrix it differs from the fp32 iteration by 1.9% in
   relative Frobenius norm (singular values [0.673, 1.137] against [0.682, 1.134]). Helen
   approved it after that number; the speed gain and the effect on recall and loss are not
   measured. Prediction: evaluation from 20 s to under 8 s per sequence, meta-training step
   from 301 s to under 150 s, recall within 0.02 of +1.0241.
6. Then: the meta-learned preconditioner as a slow weight; two Muon steps per chunk
   (evaluation only first; measure second-order memory with `scripts/memory_probe.py`); a
   reasoning form of the recall test; arm D's outer-rate sweep at `truncate_bptt=1`; the
   forgetting probe on Llama; seeds; bring `docs/preprint/main.tex` up to date.

Not planned: a KV cache beyond the window. It would answer the recall test on its own and make
the method a hybrid; it is kept only as a possible baseline (`literature.md`, "Cache hybrids").

<!-- 2.4e-4 is the inner rate of job 14247913, taken from its sbatch command, not from a recorded result -->
<!-- derived: 2.4e-4 -->

<!-- derived: 0.5 -->
