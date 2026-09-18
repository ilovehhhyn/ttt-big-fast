#!/bin/bash
# Arm C hyperparameter sweep, run in the plan's staged order so each stage fixes the
# previous winner instead of exploding into a full grid (plan Task 9).
#
#   stage 1  inner LR x optimizer   (6 runs)   -> pick eta, pick normalized_sgd vs adamw
#   stage 2  outer LR               (6 runs)   -> at the stage-1 winner
#   stage 3  LoRA rank              (2 runs)   -> 16 and 256; 64 already done in stage 1
#   stage 4  delta decay lambda     (2 runs)   -> 0.05 and 0.2; 0 already done
#
# Usage: ./sweep_C.sh <stage> [extra args forwarded to ttt.run]
set -euo pipefail
STAGE=${1:?stage 1-4}; shift || true
DATA=/scratch/gpfs/ARORA/hh9077/data/dclm8k
COMMON="--arm C --mode train --data $DATA --steps 250 --eval-sequences 32 --remat-blocks"
submit () { sbatch --job-name="$1" --qos=gpu-medium --time=12:00:00 --parsable \
            scripts/della/run_arm.sbatch $COMMON --out "results/$1.json" "${@:2}"; }

case "$STAGE" in
  1) for opt in normalized_sgd adamw; do for eta in 3e-4 1e-3 3e-3; do
       submit "C_s1_${opt}_eta${eta}" --inner "$opt" --inner-lr "$eta" --outer-lr 1e-3 --lora-rank 64
     done; done ;;
  2) for olr in 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2; do
       submit "C_s2_olr${olr}" --inner "${INNER:?set INNER}" --inner-lr "${ETA:?set ETA}" --outer-lr "$olr" --lora-rank 64
     done ;;
  3) for r in 16 256; do
       submit "C_s3_r${r}" --inner "${INNER:?}" --inner-lr "${ETA:?}" --outer-lr "${OLR:?}" --lora-rank "$r"
     done ;;
  4) for lam in 0.05 0.2; do
       submit "C_s4_lam${lam}" --inner "${INNER:?}" --inner-lr "${ETA:?}" --outer-lr "${OLR:?}" --lora-rank "${RANK:?}" --delta-decay "$lam"
     done ;;
  *) echo "unknown stage $STAGE" >&2; exit 1 ;;
esac
