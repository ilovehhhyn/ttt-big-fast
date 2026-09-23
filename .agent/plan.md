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
| F | a small extra MLP (paper layout) | as C | NOT IMPLEMENTED; `--arm F` refuses to run |

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
   there grew from +0.0015 (10 steps, 17/22) to +0.0058 (20 steps, 22/22); the 60- and 150-step
   2x2 will say whether it keeps growing.

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
| 14330258, 14330259 | `C32k_ctl60` (2 GPUs), `C32k_ctl150` (4 GPUs): `--inner none` controls for the 60- and 150-step runs | `gpu-test`, 1 h | the 2x2 at 60 and 150 steps |
| 14330856, 14330857 | `muonval_t4`, `muonval_t2`: 6 steps of arm C through Muon at 1.2e-4 (5 steps also rounds the 10% warmup to zero) | `gpu-test`, 1 h | memory and speed of meta-training through Muon |

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

1. If `muonval_t4` (job 14330856) fit and took under 300 s per step: submit 40 steps of arm C with
   `--inner muon --inner-lr 1.2e-4`, `--eval-ttt-off`, and the `--inner none` control (the
   existing `C_32k_k1024_ctl_s40` control was trained with the same outer settings and can be
   reused: its weights do not depend on the inner rule). Use a chain of 1-hour `gpu-test`
   links (`scripts/della/submit_chain.sh` with `SBATCH_EXTRA="--qos=gpu-test"`) if one link is
   not enough; sizes come from the validation job's `sec_per_step`.
2. Score both sets of weights with Muon at 1.2e-4: recall (`scripts/recall_probe.py`,
   `--gap 17408 --pairs 32`) and the 2x2 (`scripts/two_by_two.py`).
3. Prediction, written 2026-09-23 before the run: loss at or below 2.6777 with TTT on; recall
   at or above +1.0241; interaction at least twice +0.0083. If recall stays near +1.0241 and
   the interaction near +0.0083, meta-learning adds nothing to a strong fixed write rule, and
   the thesis is not supported at this scale.

## Planned after that, in order

1. Speed up Muon's Newton-Schulz iteration (bf16 or TF32; measured 20 s per sequence pass
   against 3.4 on an A100). Confirm the recall numbers do not change.
2. Make the preconditioner a slow weight (a low-rank matrix per fast layer that reshapes the
   keys before each write) and meta-learn it. This is the "LoRA weights the fast update" idea.
3. Two Muon steps per chunk (evaluation only first): the sub-agents measured 2x recall in a
   small model. Second-order memory grows with the number of steps.
4. A reasoning form of the recall test (facts stated early, a question much later whose answer
   is not in the text). Copying is the necessary first step; nothing is built yet.
5. Arm D's outer-rate sweep at `truncate_bptt=1`; arm F; the forgetting probe on Llama; seeds;
   bring `docs/preprint/main.tex` up to date (it still describes the JAX plan).

Not planned: a KV cache beyond the window. It would answer the recall test on its own and make
the method a hybrid; it is kept only as a possible baseline (`literature.md`, "Cache hybrids").

<!-- 2.4e-4 is the inner rate of job 14247913, taken from its sbatch command, not from a recorded result -->
<!-- derived: 2.4e-4 -->
