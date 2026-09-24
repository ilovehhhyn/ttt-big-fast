# Operations: Della, scripts, and the mistakes that already cost a job

This file tells the next agent how to run things without repeating the failures of the first
week. The cluster rules are hard rules; each one is an incident.

## Access

- Host alias `della-pli` (login node with one H100 PCIe). Scripted access rides Helen's SSH
  ControlMaster socket. When a command returns `Connection closed by UNKNOWN port 65535` or the
  socket is missing (`ssh -O check della-pli`), the session has lapsed: ask Helen to sign in.
  Never enter her password.
- Everything lives under `/scratch/gpfs/ARORA/hh9077`: repo mirror `ttt-big-fast`, `results`,
  `logs`, `data`, `hf` (model cache), `e2e` (reference clone).
- Sync code with `git pull --ff-only` on Della after pushing from the laptop. Della has no
  internet on compute nodes; set `HF_HUB_OFFLINE=1` in jobs (the sbatch scripts do).
- Interpreters: `/opt/anaconda3/bin/python3` on the laptop, `.venv/bin/python` with
  `PYTHONPATH=$PWD` on Della. Plain `python3` fell back to a 3.9 once and broke the suite.
- `.venv/bin/python` is a SYMLINK to a uv-managed interpreter. On 2026-09-24 16:30 that
  interpreter moved from `/home/hh9077/.local/share/uv/python/` to
  `/scratch/gpfs/ARORA/hh9077/uv/python/cpython-3.12.14-linux-x86_64-gnu/`, and every job
  failed in one second with `execve(): .venv/bin/python: No such file or directory`. The link
  and `home =` in `.venv/pyvenv.cfg` now point at the scratch path (old cfg kept as
  `pyvenv.cfg.bak-20260924`). Before a long queue of jobs, run
  `.venv/bin/python -c 'import torch'` on the login node.

## The login node

- A watchdog kills GPU processes after about 15 minutes and CPU-heavy processes after about
  10 CPU-minutes (exit 137), sometimes taking the SSH session with it.
- Use it only for evaluation runs of a few minutes, launched DETACHED:
  `( nohup scripts/della/<script>.sh > /scratch/gpfs/ARORA/hh9077/logs/<name>.log 2>&1 < /dev/null & )`.
  Each login script runs one short process at a time under `timeout -s KILL 780`, skips finished
  work, and waits for other login scripts by `pgrep` (a `pgrep -f NAME || launch` on the SAME ssh
  command line matches itself and skips the launch; launch unguarded).
- Never run the test suite or uncapped CPU PyTorch there. Cap threads: `OMP_NUM_THREADS=4`.
- A Muon evaluation of 32 sequences with `--eval-ttt-off` does NOT fit the 13-minute limit
  (killed at 780 s on 2026-09-24, job `cell_k1024_plainft_40_muon`; the fp32 Newton-Schulz
  rounds are not faster on the login H100 PCIe). Muon evaluations and every recall test go
  through `recall_muon.sbatch` on `gpu-test`.
- Never hold an SSH session open with `until`/`sleep` loops. Poll with short commands. Put any
  wait-then-act sequence inside a detached script on Della.

## Slurm

| fact | value |
|---|---|
| account, QOS for short jobs | `--account=arora`; `--qos=gpu-test`: at most 61 minutes, 3 jobs per user (a 4th pends with `QOSMaxJobsPerUserLimit`), starts within minutes |
| long jobs on `arora` | wait about 5 to 7 days; a dependent chain link ages only after its predecessor ends |
| long jobs on PLI (since 2026-09-23) | `--account=pli_x --partition=pli --qos=pli-low`: 38 nodes of 8 H100s, 15-day limit, no preemption (`PreemptMode=OFF`), at most 16 GPUs at once per user, priority 0 among PLI jobs; a 4-GPU 48-hour job was estimated to start the next evening. `pli-lc` is the same with a 3-day limit. Pass these on the command line or through `SBATCH_EXTRA` for `submit_chain.sh`; they override the `#SBATCH --account=arora` header |
| GPU memory | always `--constraint=gpu80` (the partition mixes 40 and 80 GiB A100s); in every sbatch script |
| speed | a batch A100 is about 2.1x slower than the login H100; size wall time from a BATCH node and include model load (about 1 min) and evaluation (about 4 min per 32 sequences at 32K, doubled by `--eval-ttt-off`, 6x with Muon) |
| multi-GPU | `scripts/della/run_arm_ddp.sbatch` with `srun --wait=0` (srun otherwise kills the remaining ranks 60 s after rank 0's peers exit) |
| resume | every training run checkpoints each step (`--ckpt-every`), fingerprints its settings, and resumes exactly; `scripts/della/submit_chain.sh NAME LINKS HOURS ARGS...` submits a chain with `afterany` |
| a long evaluation | must write results in pieces (`context_value.py --skip-sequences`, `cv_128k.sbatch <skip> <count>`): one job scored 5 of 12 sequences in 31 minutes and saved nothing |

## Before every sbatch (the checklist)

1. Critique the diff: config combinations no test exercises, values that change silently with
   another setting, tensors shared across loop iterations, other call sites.
2. Run the full suite locally and gate on the exit code, never on a piped `tail`:
   `/opt/anaconda3/bin/python3 -m pytest tests/ -x -q > LOG 2>&1; echo exit=$?`.
3. Prove a new test can fail: plant the bug, watch the test fail, restore.
4. Print the parameter counts of any new configuration and read them (`[run] fast=... slow=...`).
   An arm once had an EMPTY slow set; a placeholder arm once equalled arm C.
5. Inner rate in the unit 1/sqrt(n_fast) = 7.05e-5 for 201M fast parameters. There is no
   default: `--inner-lr` is required. 1e-3 diverges (loss 20).
6. A 10% warmup rounds to zero below 6 steps and hard-errors (Python rounds 0.5 to 0); use
   `--steps >= 6`. This caught validation jobs submitted with `--steps 3` and `--steps 5`
   (14247911, 14247912, 14330212); the error message now names the smallest count that works.
7. Run a short validation job that reaches evaluation and writes its result before any long
   job, then size the long job from its `sec_per_step`.
8. Memory: measure, do not reason. `scripts/memory_probe.py` with `--truncate-bptt`,
   `--inference`. At 32K one 80 GiB GPU holds arm C at window 8192 with truncation 2 (68 GiB
   peak) and at window 1024 with truncation 4 (42 GiB). Arm D fits only at truncation 1.

## How the main things are run

Train arm C for 40 steps at window 1024 (the setting of most round-2 numbers):

```bash
sbatch --job-name=NAME --qos=gpu-test --time=01:00:00 scripts/della/run_arm.sbatch \
  --arm C --mode train --data /scratch/gpfs/ARORA/hh9077/data/pg19_32k \
  --seq-len 32768 --chunk 1024 --window 1024 --prefix-segment 1024 --fast-blocks 4 \
  --remat-group 1 --truncate-bptt 4 --remat-blocks --dtype bf16 \
  --outer-lr 4e-4 --lora-rank 64 --steps 40 --tokens-per-step 131072 \
  --inner normalized_sgd --inner-lr 4e-6 --eval-sequences 32 --eval-ttt-off \
  --out /scratch/gpfs/ARORA/hh9077/results/NAME.json
```

Its plain fine-tuning control is the same command with `--inner none` (equivalent to
`--inner-lr 0` and 1.9x faster).

Options added on 2026-09-23 (all off by default; none has run on Llama yet):

| flag | what it does | belongs to |
|---|---|---|
| `--ns-dtype bfloat16` | the five Newton-Schulz rounds in bf16; weights, gradient and update stay fp32 | `--inner muon` only |
| `--weight-norm row_reset` | after every inner step each row of a fast matrix is rescaled to its pre-step norm (LaCT Alg. 1 and 3) | any inner rule |
| `--token-rates` | per-token learning rates on the write, one linear layer per fast block, a slow parameter (LaCT Eq. 4); eta = 1 at init | arms C and D |
| `--arm F --prime-intermediate 2048` | the paper layout: a small extra fast MLP per fast block with output RMSNorm and a zero gate; its W_0 is trained by the outer loop | arm F requires the width; other arms refuse it |
| `--chunk 2048 --window 2048` | the larger chunk (no new code); the window must be at least the chunk | any arm |

Every one of these is recorded in the result file's `args` block and in the resume fingerprint;
a checkpoint written before a flag existed resumes only at the flag's default.

Evaluate trained weights under another inner rule with
`--mode eval --load-slow RESULT.ckpt` (settings may differ; a resume goes through `--ckpt` and
must match).

The recall test on a set of weights:

```bash
.venv/bin/python scripts/recall_probe.py --arm C --lora-rank 64 --mode eval \
  --data /scratch/gpfs/ARORA/hh9077/data/pg19_32k --seq-len 32768 --chunk 1024 \
  --window 1024 --prefix-segment 1024 --fast-blocks 4 --remat-group 1 --remat-blocks --dtype bf16 \
  --inner muon --inner-lr 1.2e-4 --eval-ttt-off \
  --source-start 2048 --length 1024 --cue 32 --gap 17408 --pairs 32 \
  --load-slow /scratch/gpfs/ARORA/hh9077/results/C_32k_k1024_t4_s40.ckpt \
  --out /scratch/gpfs/ARORA/hh9077/results/recall_NAME.json
```

It exits non-zero if recall without TTT is not exactly 0 while the gap exceeds attention's
reach. The 2x2: `scripts/two_by_two.py --meta A.json --plain B.json` (both evaluated with
`--eval-ttt-off`). Paired effect of one run against a baseline file:
`scripts/paired_ttt_effect.py RESULT.json --baseline OTHER.json`. Gain against ceiling and
damage per document: `scripts/ttt_vs_context_value.py`.

## Scripts map

| script | purpose |
|---|---|
| `ttt/run.py` | the experiment runner (arms A to E, train or eval, resume, `--load-slow`, `--eval-ttt-off`) |
| `scripts/della/run_arm.sbatch`, `run_arm_ddp.sbatch`, `submit_chain.sh` | one GPU, several GPUs, chained links |
| `scripts/della/round2_start.sh` | detached orchestrator: validates 2-GPU parity, checks the corpus, submits the matched-budget chains |
| `scripts/della/login_*.sh` | detached login-node evaluation batches (recall, SlimPajama cells, preconditioned rule) |
| `scripts/della/recall_lr.sbatch`, `recall_muon.sbatch` | recall and loss against inner rate; Muon on un-tuned or trained weights |
| `scripts/context_value.py` | what old context is worth to a healthy model (`--only-label`, `--skip-sequences`) |
| `scripts/recall_probe.py`, `ttt/eval/recall.py` | the recall test |
| `scripts/key_basis.py`, `ttt/optim/key_basis.py` | shared key directions for `--inner preconditioned_sgd` |
| `scripts/memory_probe.py` | GPU memory of one sequence, with a per-chunk trace |
| `scripts/compare_runs.py` | step-by-step comparison of two runs (resume, data parallel) |
| `scripts/check_report_numbers.py` | every decimal in a report must occur in its cited sources |
| `ttt/data/prepare.py`, `scripts/della/prep_cpu.sbatch`, `extend_pg19.sbatch` | corpora; tokenize in CPU jobs, never on the login node |

## Reading results

A result JSON holds `args` (every setting), `history` (per-step loss, grad norm, lr, seconds),
`eval` and `eval_ttt_off` (loss, `per_sequence_loss`, `token_nll` by position), `peak_gib`. A
recall JSON holds `pairs`, per-condition `per_pair_recall` and `recall_by_offset`, and
`exact_floor_violations`. Quote per-document intervals, never per-sequence ones: the 32
PG-19 sequences come from 22 books.

<!-- derived: 0.5 -->
