#!/bin/bash
# Round 2 orchestrator. Runs DETACHED on the Della login node and gates every step itself, so
# that nothing depends on an SSH session staying alive:
#
#   cd /scratch/gpfs/ARORA/hh9077/ttt-big-fast
#   ( nohup scripts/della/round2_start.sh > /scratch/gpfs/ARORA/hh9077/logs/round2_start.log 2>&1 < /dev/null & )
#
# It does only light work itself (sbatch, squeue, one JSON comparison). CPU threads are capped:
# uncapped CPU PyTorch on this node was SIGKILLed on 2026-09-21 and took the SSH session with it.
set -uo pipefail
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast; RES=/scratch/gpfs/ARORA/hh9077/results
LOGS=/scratch/gpfs/ARORA/hh9077/logs; DATA=/scratch/gpfs/ARORA/hh9077/data
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH="$ROOT"
cd "$ROOT"
say() { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

# The protocol every 32K arm C run so far has used (PG-19, k = 8192, b = 1024).
C32K="--arm C --mode train --data $DATA/pg19_32k --seq-len 32768 --chunk 1024 --window 8192
      --fast-blocks 4 --remat-group 1 --prefix-segment 8192 --truncate-bptt 2 --remat-blocks --dtype bf16
      --outer-lr 4e-4 --lora-rank 64 --eval-sequences 32"

say "code: $(git log --oneline -1)"
NEED=760217600   # 725 steps x 32 sequences x 32768 tokens
# The matched run reads the EXTENDED corpus (scripts/della/extend_pg19.sh): same validation
# split byte for byte, larger training split. Until it is READY, the gate below refuses.
MATCH_DATA="$DATA/pg19_32k_full"
if [ -e "$MATCH_DATA/READY" ]; then
  HAVE=$(.venv/bin/python -c "import json;print(json.load(open('$MATCH_DATA/train.json'))['num_tokens'])")
else
  HAVE=0
fi
say "matched-budget corpus $MATCH_DATA: $HAVE train tokens ready; the run consumes $NEED"

# ---- 0. Launch the login-node work that needs no gate (both scripts skip what is done).
# Never two instances at once: both would start the same unfinished item.
for job in round2_data round2_login; do
  if pgrep -u hh9077 -f "$job.sh" > /dev/null; then say "$job.sh is already running"
  else ( nohup "scripts/della/$job.sh" >> "$LOGS/$job.log" 2>&1 < /dev/null & ); say "launched $job.sh"; fi
done

# ---- 1. What does Slurm say about the job shapes? (informational)
for shape in "--time=60:00:00 --ntasks-per-node=4 --gres=gpu:4" "--time=34:00:00 --ntasks-per-node=4 --gres=gpu:4" \
             "--time=24:00:00 --ntasks-per-node=2 --gres=gpu:2"; do
  say "test-only [$shape]: $(sbatch --test-only $shape scripts/della/run_arm_ddp.sbatch --arm C 2>&1 | grep -oE 'start at [^ ]+|error.*' | head -1)"
done

# ---- 2. Validate data parallelism on REAL hardware before trusting it with 320 GPU-hours:
#         2 GPUs x 2 sequences is the same global batch as the single-GPU run C_32k_q10, so the
#         two must agree to within the measured run-to-run noise (2.5e-4 in loss).
VAL="$RES/C_32k_ddp2_val.json"
if [ ! -e "$VAL" ]; then
  jid=$(sbatch --parsable --job-name=C32k_ddp2 --qos=gpu-test --time=01:00:00 --ntasks-per-node=2 --gres=gpu:2 \
        scripts/della/run_arm_ddp.sbatch $C32K --inner normalized_sgd --inner-lr 4e-6 --steps 10 \
        --tokens-per-step 131072 --out "$VAL")
  [[ "$jid" =~ ^[0-9]+$ ]] || { say "GATE FAILED: the 2-GPU validation was not accepted by sbatch ('$jid'). Nothing submitted."; exit 1; }
  say "submitted 2-GPU validation job $jid; waiting for it"
  while squeue -j "$jid" -h 2>/dev/null | grep -q .; do sleep 60; done
  say "validation job $jid ended: $(sacct -j "$jid" -n -X -o State,Elapsed | head -1)"
fi
if [ ! -e "$VAL" ]; then
  say "GATE FAILED: the 2-GPU validation produced no result. Matched-budget jobs NOT submitted."
  tail -5 "$LOGS"/C32k_ddp2-*.out 2>/dev/null | cut -c1-200
  exit 1
fi
.venv/bin/python scripts/compare_runs.py "$RES/C_32k_q10.json" "$VAL" --metrics loss --tol 1e-3 | tail -14
if ! .venv/bin/python scripts/compare_runs.py "$RES/C_32k_q10.json" "$VAL" --metrics loss --tol 1e-3 > /dev/null; then
  say "GATE FAILED: 2 GPUs do not reproduce the single-GPU run within 1e-3. Matched-budget jobs NOT submitted."
  exit 1
fi
say "GATE PASSED: 2 GPUs reproduce the single-GPU run within 1e-3 nats"

# ---- 3. The matched extension budget: 725 steps x 32 sequences (1,048,576 tokens per step),
#         on 4 GPUs. Arm C, and the extension-only control it must be compared with. Two links
#         each: the second resumes if the first hits its wall limit, and exits at once otherwise.
# The loader cycles when the corpus runs out, which would turn one pass over fresh text (what
# the reference does) into several epochs over a smaller set. Refuse rather than drift.
if [ "$HAVE" -lt "$NEED" ]; then
  say "GATE FAILED: $MATCH_DATA has $HAVE train tokens ready, fewer than the $NEED the matched budget"
  say "consumes. scripts/della/extend_pg19.sh prepares it and then reruns this script."
  exit 1
fi
export SBATCH_SCRIPT=scripts/della/run_arm_ddp.sbatch SBATCH_EXTRA="--ntasks-per-node=4 --gres=gpu:4"
if ! squeue -u hh9077 -h -o %j | grep -q '^C32k_match_1$'; then
  # 8 sequences per GPU per step at 33 s each = 264 s per step; 725 steps = 53 h, plus evaluation.
  scripts/della/submit_chain.sh C32k_match 2 60 $C32K --data "$MATCH_DATA" --inner normalized_sgd --inner-lr 4e-6 \
      --steps 725 --tokens-per-step 1048576 --eval-ttt-off
fi
if ! squeue -u hh9077 -h -o %j | grep -q '^C32k_match_ctl_1$'; then
  # --inner none is equivalent to --inner-lr 0 and 1.9x faster: 17.2 s per sequence -> 28 h.
  scripts/della/submit_chain.sh C32k_match_ctl 2 34 $C32K --data "$MATCH_DATA" --inner none \
      --steps 725 --tokens-per-step 1048576
fi
say "queue now:"; squeue -u hh9077 -o '%.10i %.18j %.2t %.8f %.11l %.6D %R' | grep -E 'JOBID|C32k|D32k'
say "=== round2_start finished ==="
