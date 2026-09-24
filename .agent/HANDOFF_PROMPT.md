# Prompt for a fresh agent taking over `ttt-big-fast`

Copy everything below the line into the first message of a new Claude Code session opened in
`/Users/helenhui/ttt-big-fast`. It was written on 2026-09-23 at 17:50 ET; job states after
that time must be re-read on Della.

---

You are taking over a research project from another agent. The project lead is Helen Hui
(Princeton, netid `hh9077`). Your job is to continue the experiments, keep every record
correct, and report in plain English. Read this whole message before doing anything.

## 1. Read these first, in this order

1. `.agent/README.md`: the project in one paragraph, the state in ten lines, the map of every
   document, the rules.
2. `.agent/plan.md`: hypothesis, arms, protocol, decision rules, what is queued (with job IDs),
   the next experiment and its written prediction.
3. `.agent/results.md`: every headline number with its 95% interval and the section of
   `docs/results/results.md` that holds it.
4. `.agent/operations.md`: how to run things on Della and the pre-sbatch checklist.
5. `.agent/literature.md`: the papers, what we took from each, and who checked each claim.
6. `.agent/log.md`: one entry per day since 2026-09-16.
7. `docs/results/results.md`: the canonical, dated record. Read at least the sections dated
   2026-09-21, 2026-09-22 and 2026-09-23 in full.
8. `docs/research/FINDINGS.md` sections 13 to 15: why arm C ran out of memory and the fixes;
   the cross-check against the reference implementation; why the sliding window breaks the
   un-tuned model.
9. Your memory directory (`~/.claude/projects/-Users-helenhui-ttt-big-fast/memory/`, loaded
   through `MEMORY.md`): `feedback-preflight-checklist.md`, `project-second-order-memory-lessons.md`,
   `feedback-verify-before-claiming.md`, `della-login-node-watchdog.md`,
   `project-big-fast-small-slow-ttt.md`, `feedback-plain-writing.md`. Update them as you go.
10. `~/.claude/CLAUDE.md`: Helen's working preferences. They override everything below.

Then run the test suite locally and confirm it passes before you touch code:

```bash
/opt/anaconda3/bin/python3 -m pytest tests/ -x -q > /tmp/pytest.log 2>&1; echo exit=$?
```

## 2. The project in one paragraph

TTT-E2E (arXiv 2512.23675) trains a sliding-window language model whose last-quarter MLPs are
"fast weights": at test time they take one gradient step per 1024-token chunk on that chunk's
next-token loss, so the model keeps learning from what it reads; all other parameters are
meta-learned through that inner loop. This project inverts the sizes: the fast weights are the
FULL pretrained MLP matrices of the last 4 of Llama-3.2-1B's 16 blocks (201,326,592
parameters) and the slow, meta-learned weights are small: a rank-64 LoRA on `wq, wk, wv, wo, w1,
w2, w3`, the RMSNorm gains and one learned log step size per fast tensor (45,156,364
parameters). The thesis (H1) is that this small slow set, trained through the inner loop,
captures most of what test-time training can give. The measurable form is a 2x2: slow weights
trained {through the inner loop, by plain fine-tuning} x evaluated {TTT on, TTT off}; the
interaction is what H1 claims. Code is PyTorch in `ttt/`; experiments run on Princeton Della.

## 3. What is known (all numbers are nats per token, measured, with intervals in `.agent/results.md`)

1. The window breaks the pretrained model. At 32K on PG-19: full attention 2.3092, window 8192
   3.7119, window 1024 4.9895. Old context is worth only +0.0208 to a healthy model at window
   8192 (`scripts/context_value.py`), +0.0992 at 1024; on SlimPajama arXiv +0.2548 at 1024.
   TTT alone gains +0.1405 at window 8192, so at least 85% of that is repair, not memory.
   Never call a gain "memory" without this comparison.
2. The thesis is weakly supported. Interaction: +0.0015 (window 8192, 10 steps, 17/22 books),
   +0.0058 (8192, 20 steps, 22/22), +0.0126 then +0.0083 (1024, 10 and 40 steps, 22/22), +0.0074
   (SlimPajama, 40 steps, 85/88 documents). Positive everywhere, about 0.3% of the loss.
3. Memory measured directly (`scripts/recall_probe.py`: plant a 1024-token passage, repeat it
   17,408 tokens later, past attention's reach of 16 x 1023 tokens; recall = loss drop on the
   repeat; exactly 0 without TTT, checked on every run): the original write (normalized SGD at
   4e-6) stores +0.0717 un-tuned and +0.1054 on the 40-step weights, against +2.6920 with full
   attention: 4%. Recall is flat along the passage, so the write is the limit, not the read.
4. Why the write is weak: the chunk gradient G = sum_t d_t k_t^T is dominated by a few input
   ("key") directions shared by all tokens (one direction carries 32 to 60% of the key energy
   in each fast block), which caps the step. Two rules that spread the update evenly fix it.
   Muon (`--inner muon`, all singular values set to 1) at 1.2e-4, thirty times the old step,
   recalls +1.0241 (38% of full attention) at loss 2.6895 against 2.6777; at 2.4e-4, +1.5005
   (56%) at 2.7937. The preconditioned rule (`--inner preconditioned_sgd`, shared key
   directions removed, `scripts/key_basis.py`) reaches +0.7025 at 2.7648 and costs nothing.
5. Learning rates. Inner: the unit is 1/sqrt(n_fast) = 7.05e-5; the loss optimum is 4e-6 to 7e-6
   on every window and dataset; memory wants more (un-tuned recall peaks at 1.4e-5); training
   AT the larger rate keeps the memory at little loss cost (trained at 1e-5: recall +0.2931,
   loss 2.6820; at 2e-5: +0.5201, 2.7243). AdamW as the inner rule is worse than normalized SGD.
   Outer: 4e-4 for the LoRA; all-weights-slow (arm D) diverges at 4e-4 and reaches 2.7306 at 4e-5.
6. At the reference window, longer LoRA training shrinks what TTT adds on the same weights:
   +0.0248 (8 steps), +0.0125 (60), +0.0067 (150), all under the +0.0208 ceiling. 150 steps of
   4 sequences (2.4664) beat 20 steps of 32 (2.5664) at equal tokens. Truncating the
   meta-gradient to 1 chunk instead of 2 changes nothing (-0.0004 per book).
7. We have NOT matched TTT-E2E's headline (parity with full attention at 32K after 725 steps
   of 32 sequences). Our runs used 300 to 600x less training. The matched-budget runs are
   queued on the PLI partition.
8. Decided by Helen: no KV cache beyond the window in the method (recall would come from the
   cache, not the fast weights). The single method is: fast MLPs written by an equalized
   gradient step (Muon), slow weights meta-learned through it.

## 4. What is running or queued right now (2026-09-23 17:50; re-read with `squeue -u hh9077`)

| jobs | what | when it lands, do this |
|---|---|---|
| 14333213 to 14333217 | `C_32k_k1024_muon_s40`: 40 steps of arm C meta-trained THROUGH Muon at 1.2e-4, window 1024, truncation 4, `--eval-ttt-off`; chain of five 1-hour `gpu-test` links; 286 s per step | the most important run. When `results/C_32k_k1024_muon_s40.json` exists: (a) evaluate the plain control `C_32k_k1024_ctl_s40.ckpt` with `--inner muon --inner-lr 1.2e-4 --eval-ttt-off --load-slow` (login GPU, about 12 min) into `results/cell_k1024_plainft_40_muon.json`; (b) `scripts/two_by_two.py --meta results/C_32k_k1024_muon_s40.json --plain results/cell_k1024_plainft_40_muon.json`; (c) recall of both weight sets with Muon 1.2e-4 (`sbatch --qos=gpu-test scripts/della/recall_muon.sbatch 1.2e-4 C_32k_k1024_muon_s40`). Prediction written before the run: loss at or below 2.6777, recall at or above +1.0241, interaction at least twice +0.0083. If recall stays near +1.0241 and the interaction near +0.0083, meta-learning adds nothing to a strong fixed write rule. |
| 14330259 | `C32k_ctl150`: plain 150-step control at window 8192, 4 GPUs, `gpu-test` | `scripts/della/login_cells_k8192_s60_s150.sh` is already running detached and waits for it; read `logs/login_cells_k8192_s60_s150.log` for the 60- and 150-step 2x2. |
| 14169729 | `C32k_bs32s60`: 60 steps of 32 sequences, window 8192, single A100, running since 09:55 (about 18 h) | record its loss and TTT on/off per book (`scripts/paired_ttt_effect.py results/C_32k_bs32s60.json`); it has no plain control yet. |
| 14330761 to 14330768 | the matched budget (725 x 32) on PLI: `C32k_match` + `C32k_match_ctl` (window 8192), `Ck1024_match` + `Ck1024_match_ctl` (window 1024, truncation 4); 4 H100s each; second links are spares that resume or exit | estimated start 2026-09-24 11:00 to 14:00. When each pair finishes: 2x2 via `--load-slow` on the control's checkpoint, paired per book, and read the beyond-window loss against the healthy level (about 2.33 at window 8192). These use the OLD write rule (normalized SGD 4e-6) on purpose: they test the thesis at the reference budget with the weak memory. |
| 14330857 | `muonval_t2`: 6-step Muon validation at truncation 2 (250 s per step) | informational only; `muonval_t4` already showed truncation 4 fits (peak 60.4 GiB). |

Poll with short commands only, for example:

```bash
ssh della-pli "squeue -u hh9077 -h -o '%.10i %.22j %.2t %.9M %.10P %R' | grep -v klr-"
ssh della-pli "grep -E '^\[eval\]|^ttt_on|^\[train\]' /scratch/gpfs/ARORA/hh9077/logs/<name>-<job>.out | tail -5"
```

## 5. After those, in this order (the plan; details in `.agent/plan.md`)

1. Speed up Muon's Newton-Schulz iteration (`ttt/optim/inner.py`, `newton_schulz5` runs in
   float32; try TF32 or bf16 for the iteration only, keep the fast weights fp32). Confirm on
   the recall test that the numbers do not change. Meta-training through Muon costs 286 s per
   step against 55 for normalized SGD; the matched budget through Muon is not affordable until
   this is done.
2. Make the write preconditioner a slow weight: a low-rank matrix per fast layer that reshapes
   the keys before each write, meta-learned through the inner loop. This is Helen's idea
   ("the LoRA weights the fast update"). Start from `PreconditionedSGD` and the key basis;
   the basis becomes a parameter in the slow set (`ttt/model/naming.py` decides slow membership).
3. Two Muon steps per chunk, evaluation only first (a loop in `_chunk_step` in
   `ttt/train/inner_loop.py`, reported loss stays the loss before the first step). Measure
   second-order memory with `scripts/memory_probe.py` before any training run.
4. A reasoning form of the recall test (facts stated early, a question much later whose answer
   is not in the text). Nothing is built. Copying is the necessary first step and is measured.
5. Matched-budget run through Muon once 1 is done; the PLI partition can hold it.
6. Housekeeping: arm D outer-rate sweep at `truncate_bptt=1`; arm F was built on
   2026-09-23 (`--arm F --prime-intermediate 2048`) but has never run; the forgetting probe on
   Llama; seeds; `docs/preprint/main.tex` still describes the JAX plan.

Before every long job: a short validation job that reaches evaluation and writes its result,
sized from a batch node's `sec_per_step`, with a checkpoint every step and a spare chain link.

## 6. Rules that are not negotiable

- Free data and tools only. All cluster work stays under `/scratch/gpfs/ARORA/hh9077`.
- Never enter Helen's password. Never use a token pasted into chat. When SSH returns
  `Connection closed by UNKNOWN port 65535` or `ssh -O check della-pli` finds no socket, the
  session lapsed: tell Helen and ask her to sign in. Do not retry in a loop.
- Load the `mdx` skill (`~/.claude/skills/mdx/SKILL.md`) before writing or editing any code,
  test, commit message, doc or script, every time. Read its `references/testing.md` before a
  test file and `references/prose.md` before a document.
- No silent fallbacks: a setting that cannot be honoured is a hard error at construction, with
  the fix named in the message. Keep one explicit opt-out value.
- Isolate one factor by holding everything else identical (same trained weights, switch on and
  off through `--eval-ttt-off` or `--load-slow`). Two separately trained runs never decompose
  into additive contributions.
- Samples from one document are not independent: cluster paired differences by document
  (`ttt/eval/paired.py`, `scripts/paired_ttt_effect.py`, `scripts/two_by_two.py`) before
  quoting an interval. The 32 PG-19 sequences come from 22 books.
- State a prediction before a result comes back. Run the control that could undercut a
  headline number. Separate memory from repair with `scripts/context_value.py` and the recall
  test.
- Never state an inference as a finding. Open the result file; do not trust a caption.
- Write in plain textbook English: short sentences, standard terms, first sentence says what a
  thing is and what it is for, no invented shorthand, no dashes, short messages that lead with
  the answer and the key numbers. Helen sometimes invokes `/i-have-adhd` and `/humanizer`.
- Commit with `type(scope): imperative subject`, and end every commit message with
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` (replace the model name with
  yours). Push to `origin main`, then `git pull --ff-only` on Della. Run the full suite and gate
  on its exit code before any push that touches code.

## 7. Mistakes that already cost a job or a wrong number (do not repeat them)

Cluster and jobs:
- The login node (`della-pli`) kills GPU processes after about 15 minutes and CPU-heavy
  processes after about 10 CPU-minutes (exit 137), sometimes with the SSH session. Use it only
  for evaluation runs under 13 minutes, launched detached:
  `( nohup scripts/della/X.sh > /scratch/gpfs/ARORA/hh9077/logs/X.log 2>&1 < /dev/null & )`.
  Never run `pytest` or uncapped CPU PyTorch there; cap with `OMP_NUM_THREADS=4`.
- Never hold an SSH session open with `until`/`while`/`sleep` loops; three of them killed the
  ControlMaster and cost Helen a re-login. Poll with commands that return at once. Put any
  wait-then-act sequence inside a detached script on Della that gates itself and logs.
- `pgrep -f NAME || launch` on the same ssh command line matches its own command line and
  skips the launch silently. `pkill -f NAME` inside `ssh host "...NAME..."` kills its own shell.
- Always `--constraint=gpu80` (the `gpu` partition mixes 40 and 80 GiB cards); it is in every
  sbatch script. Size wall time from a BATCH node: an A100 is 2.1x slower than the login H100.
- `gpu-test` (`--qos=gpu-test`): at most 61 minutes, 3 jobs per user, starts in minutes. Long
  `arora` jobs wait 5 to 7 days. PLI (`--account=pli_x --partition=pli --qos=pli-low`): H100s,
  15-day limit, no preemption, at most 16 GPUs at once, about a day's wait; pass the options on
  the command line or through `SBATCH_EXTRA` for `submit_chain.sh`.
- Multi-GPU jobs need `srun --wait=0` (already in `run_arm_ddp.sbatch`): srun otherwise kills
  the remaining ranks 60 s after the others exit, which killed a run during evaluation.
- A job that writes its result only at the end must be sized from a measured per-item time or
  split into pieces that each write their own file (`context_value.py --skip-sequences`,
  `cv_128k.sbatch <skip> <count>`). One evaluation scored 5 of 12 sequences and saved nothing.
- A 10% warmup rounds to zero below 6 steps and hard-errors (Python rounds 0.5 to 0): use
  `--steps >= 6`. Three validation jobs were lost to `--steps 3` and `--steps 5`.
- `--inner-lr` is required whenever an inner rule is active; there is no default. 1e-3 (the
  paper's value for its 11.5M-parameter MLP) diverges here: loss 20.
- Tokenize corpora in CPU Slurm jobs (`scripts/della/prep_cpu.sbatch`), never on the login
  node (killed three times). Compute nodes have no internet: `HF_HUB_OFFLINE=1`.
- Never decide success through `|| true` or a piped `tail`; gate on the real exit code.
- `python3` on the laptop once fell back to the system 3.9; use `/opt/anaconda3/bin/python3`.
  On Della use `.venv/bin/python` with `PYTHONPATH=$PWD`.

Memory (GPU) and the second-order inner loop (`docs/research/FINDINGS.md` section 13):
- Measure memory, do not reason about it: four diagnoses were wrong until traced. Use
  `scripts/memory_probe.py --truncate-bptt N --inference --window K`, on a clean process.
  Quote the trainer's peak (68 GiB at window 8192, truncation 2; 42.0 GiB at window 1024,
  truncation 4; 60.4 GiB with Muon at truncation 4), not the probe's.
- `remat_group` must be 1: `torch.utils.checkpoint` cannot discard a graph built with
  `create_graph=True` inside the region, so bigger groups hold MORE graph.
- Truncated backpropagation only frees memory with a per-window `backward()` (implemented in
  `TTTInnerLoop.run_sequence` through `backward_scale`).
- Any NON-LEAF tensor created outside the chunk loop and read inside it is freed by the first
  window's backward ("Trying to backward through the graph a second time"). It hit twice
  (`prefix_out`, then `exp(inner_lr_log)`). Cut such tensors at `x.detach().requires_grad_(True)`,
  accumulate `.grad`, push it through the original graph once at the end. When you touch the
  loop, enumerate every outside tensor.
- Evaluation must use `inference=True` (no `create_graph`, no checkpoint regions, prefix under
  `no_grad`); it is value-identical to the training path and tested to be.
- Every flag that gates a code path belongs in the test matrix (optimizer x learned-LR x
  truncation x inference). Two bugs reached the cluster because fixtures always passed one
  optimizer or `learned_lr=False`.

Reading results:
- `squeue %M` shows elapsed time: `0:25` is 25 seconds, not minutes.
- `token_nll` in a result file is averaged over sequences by position; per-sequence numbers
  are `per_sequence_loss`. Check an array's shape and meaning before computing on it.
- Before quoting a trend, confirm every row shares every setting except the one varied; open
  the result files. A wrong 8K/16K/32K trend was quoted to Helen three times this way.
- A placeholder configuration must refuse to run (arm F did until 2026-09-23). An arm once had an EMPTY slow set
  because `"**"` was matched as a substring; print and read the parameter counts of any new
  configuration (`[run] fast=... slow=... frozen=...`).
- A signature supports a mechanism only if rival explanations predict something different:
  "gain only beyond the window" is produced by window repair as well as by memory.

## 8. Where to look when stuck

| problem | look at |
|---|---|
| how the inner loop works, where the meta-gradient is cut | `ttt/train/inner_loop.py` (`run_sequence`, `_chunk_step`, `_first_grad`), `tests/test_inner_loop.py` |
| the inner update rules and their unit convention | `ttt/optim/inner.py` (module docstring), `tests/test_inner_optim.py`, `tests/test_preconditioned_sgd.py` |
| which parameters are fast, slow, frozen | `ttt/model/naming.py`, `ttt/config.py` (`InnerConfig`, `TrainConfig`) |
| the recall test | `ttt/eval/recall.py` (docstring has the design and the floor argument), `scripts/recall_probe.py`, `tests/test_recall.py` |
| paired statistics, 2x2, regressions | `ttt/eval/paired.py`, `scripts/two_by_two.py`, `scripts/ttt_vs_context_value.py` |
| resume, fingerprints, `--load-slow` | `ttt/train/checkpoint.py`, `tests/test_checkpoint.py` |
| data parallelism by hand | `ttt/train/distributed.py`, `scripts/della/run_arm_ddp.sbatch`, `tests/test_distributed.py` |
| an OOM or a slow step | `docs/research/FINDINGS.md` section 13, `scripts/memory_probe.py`, memory file `project-second-order-memory-lessons.md` |
| what a job did | `/scratch/gpfs/ARORA/hh9077/logs/<jobname>-<jobid>.out`; `sacct -j <id> -X -o JobName,State,Elapsed` |
| a result's settings | the `args` block of its JSON in `/scratch/gpfs/ARORA/hh9077/results/` |
| the reference implementation | `/scratch/gpfs/ARORA/hh9077/e2e` on Della: `configs/experiment/*/extension/*.yaml` (the real settings; the dataclass defaults are not what they ran), `ttt/optimizers.py`, `ttt/model/loss.py` |
| whether a number in a report is real | `scripts/check_report_numbers.py REPORT.md docs/results/results.md docs/research/FINDINGS.md` |
| whether two runs agree | `scripts/compare_runs.py A.json B.json --metrics loss --tol 1e-3` |
| the corpora | `/scratch/gpfs/ARORA/hh9077/data/{pg19_32k,pg19_32k_full,pg19_128k,slimpajama_32k}/{train,val}.json` |

## 9. Skills to use

- `mdx` for all code and docs (always).
- `superpowers:brainstorming` before designing anything new; `superpowers:writing-plans` and
  `superpowers:executing-plans` for multi-step work; `superpowers:test-driven-development`;
  `superpowers:systematic-debugging` for any failure; `superpowers:verification-before-completion`
  before claiming something is done.
- `anthropic-skills:i-have-adhd` and `humanizer:humanizer` when Helen asks for a summary or a
  message; `academic-research-skills:deep-research` for literature questions (open the sources;
  never cite a number you did not read).
- Sub-agents (the `Agent` tool) for parallel, read-only investigation; give them the rules in
  section 6, tell them not to hallucinate and not to touch the cluster or the repository.

## 10. How to report to Helen

Lead with the answer. Give the key numbers with their intervals and the count of documents
in favour. Say what the result does not show. Name the next step. Keep it short. When a
result is confounded, say so and name the control that resolves it. When you made a mistake,
say so plainly and record it in the memory files and in `.agent/log.md`.

Your first message to Helen should say what you read, the current job states you observed,
and what you are about to do, in under 150 words.
