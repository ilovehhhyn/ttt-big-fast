#!/bin/bash
#SBATCH --account=arora
#SBATCH --job-name=resume-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/ARORA/hh9077/logs/%x-%j.out
# Real-hardware check of checkpoint/resume: run a training command, SIGKILL it after
# KILL_AFTER_STEPS checkpoints have been written, then run the identical command again
# and let it resume to completion. Compare the result against an uninterrupted run of
# the same configuration with scripts/compare_runs.py.
#
#   sbatch scripts/della/resume_check.sbatch <out.json> <ttt.run args...>
set -euo pipefail
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast
export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
KILL_AFTER_STEPS=${KILL_AFTER_STEPS:-4}
cd "$ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

OUT=$1; shift
CKPT="${OUT%.json}.ckpt"
rm -f "$OUT" "$CKPT"   # a stale checkpoint would turn the first leg into a resume

# Leg 1: python is launched directly (not through srun) so the SIGKILL lands on the
# training process itself: no handlers, no cleanup, exactly like a node failure.
.venv/bin/python -u -m ttt.run --out "$OUT" "$@" &
PID=$!
writes=0; last=""
while kill -0 "$PID" 2>/dev/null; do
  now=$(stat -c %Y.%s "$CKPT" 2>/dev/null || true)   # a new mtime = one more step banked
  if [ -n "$now" ] && [ "$now" != "$last" ]; then writes=$((writes + 1)); last=$now; echo "[check] checkpoint write #$writes"; fi
  if [ "$writes" -ge "$KILL_AFTER_STEPS" ]; then kill -9 "$PID"; echo "[check] SIGKILL after $writes checkpoint writes"; break; fi
  sleep 1
done
wait "$PID" || echo "[check] leg 1 exit code $? (137 = killed, as intended)"
[ ! -e "$OUT" ] || { echo "[check] FAIL: leg 1 finished before it could be killed"; exit 1; }
ls -la "$CKPT"

# Leg 2: the identical command. It must find the checkpoint and continue.
.venv/bin/python -u -m ttt.run --out "$OUT" "$@"
echo "[check] resumed run completed: $OUT"
