#!/bin/bash
# Evaluation-only runs for the Della LOGIN node, one short process at a time.
#
# The login node kills a GPU process after roughly 15 minutes, silently, so this runner is
# only for evaluations of a few minutes each (no training). Every evaluation is its own
# process; a failure is recorded and reported at the end -- never swallowed -- and a
# result that already exists is skipped, so the script can simply be rerun.
#
#   nohup scripts/della/login_evals.sh > /scratch/gpfs/ARORA/hh9077/logs/login_evals.log 2>&1 &
set -uo pipefail   # no -e: one failed evaluation must not stop the rest, but it IS reported
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast
RES=/scratch/gpfs/ARORA/hh9077/results
LOGS=/scratch/gpfs/ARORA/hh9077/logs
export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$ROOT"
MAX_SECONDS=720    # hard stop well inside the watchdog window; an eval this slow belongs on Slurm
cd "$ROOT"

# The 32K protocol shared by every run below (PG-19, k=8192, b=1024, same 32 sequences).
COMMON="--mode eval --data /scratch/gpfs/ARORA/hh9077/data/pg19_32k --seq-len 32768 --chunk 1024
        --window 8192 --fast-blocks 4 --remat-group 1 --prefix-segment 8192 --remat-blocks
        --dtype bf16 --eval-sequences 32"

failed=(); ran=0; skipped=0
run_eval() {   # run_eval <name> <ttt.run args...>
  local name=$1; shift
  local out="$RES/$name.json"
  if [ -e "$out" ]; then echo "[skip] $name (result exists)"; skipped=$((skipped + 1)); return; fi
  echo "[start] $name  $(date +%H:%M:%S)"
  local t0=$SECONDS
  timeout -s KILL "$MAX_SECONDS" .venv/bin/python -u -m ttt.run $COMMON --out "$out" "$@" > "$LOGS/login_$name.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -e "$out" ]; then
    echo "[done]  $name  $((SECONDS - t0))s  $(grep -E '^\[eval\]' "$LOGS/login_$name.log" | tail -1)"
    ran=$((ran + 1))
  else
    echo "[FAIL]  $name  exit=$rc after $((SECONDS - t0))s  -> $LOGS/login_$name.log"
    failed+=("$name")
  fi
}

# 1. Arms A and B again, now recording per-sequence losses, so the TTT-alone headline
#    (B - A) gets the same document-clustered interval as the arm C ablation.
run_eval A_32k_perseq --arm A
run_eval B_32k_perseq --arm B --inner normalized_sgd --inner-lr 4e-6

# 2. AdamW as the inner optimizer at 32K (both inner optimizers are required; only
#    normalized SGD has a 32K curve). Adam's update RMS is ~lr, the same per-element unit
#    as normalized SGD's lr_rms, so the grid brackets that optimum (4e-6).
for lr in 1e-6 2e-6 4e-6 7e-6 2e-5; do
  run_eval "B_32k_adamw_lr$lr" --arm B --inner adamw --inner-lr "$lr" --adam-eps 1e-8
done

echo "=== login_evals: ran=$ran skipped=$skipped failed=${#failed[@]} ${failed[*]:-} ==="
[ ${#failed[@]} -eq 0 ]
