#!/bin/bash
#SBATCH --account=arora
#SBATCH --job-name=recall_lr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/ARORA/hh9077/logs/%x-%j.out
# How does recall change with the inner learning rate, and what does that cost in loss?
# PG-19, 32K tokens, window 1024, the same 32 sequence pairs as scripts/della/login_recall_k1024.sh
# (already measured there: un-tuned 4e-6 -> +0.0717, 2e-5 -> +0.1348, 5e-5 unstable).
#   sbatch scripts/della/recall_lr.sbatch 1    # un-tuned model: recall and loss at more rates
#   sbatch scripts/della/recall_lr.sbatch 2    # 40-step trained weights, evaluated at higher rates
# Evaluation only. Measured on the login H100: 133 s per recall run; an A100 is about 2.1x slower,
# so each part is about 30 minutes. Finished runs are skipped, so a part can be resubmitted.
set -uo pipefail
PART=${1:?usage: recall_lr.sbatch <1|2>}
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast; RES=/scratch/gpfs/ARORA/hh9077/results; DATA=/scratch/gpfs/ARORA/hh9077/data/pg19_32k
export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$ROOT"
cd "$ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
failed=0
MODEL="--mode eval --data $DATA --seq-len 32768 --chunk 1024 --window 1024 --prefix-segment 1024 --fast-blocks 4
       --remat-group 1 --remat-blocks --dtype bf16 --inner normalized_sgd"
PROBE="--source-start 2048 --length 1024 --cue 32 --gap 17408 --pairs 32"
run() {   # run <result name> <script and its arguments...>
  local name=$1; shift
  [ -e "$RES/$name.json" ] && { echo "[skip] $name"; return; }
  echo "[start] $name $(date +%H:%M:%S)"
  .venv/bin/python -u "$@" --out "$RES/$name.json"; local rc=$?
  [ $rc -eq 0 ] || { echo "[FAIL]  $name exit=$rc"; failed=$((failed + 1)); }
}
case "$PART" in
  1) for lr in 7e-6 1e-5 1.4e-5 3e-5; do
       run "recall_pg19_k1024_untuned_lr$lr" scripts/recall_probe.py --arm B $MODEL $PROBE --inner-lr "$lr"
     done
     for lr in 1e-5 1.4e-5 3e-5; do   # loss on ordinary text at the same rates (2e-6 .. 5e-5 exist already)
       run "B_32k_k1024_lr$lr" -m ttt.run --arm B $MODEL --eval-sequences 32 --inner-lr "$lr"
     done ;;
  2) META="$RES/C_32k_k1024_t4_s40.ckpt"; PLAIN="$RES/C_32k_k1024_ctl_s40.ckpt"
     for lr in 1e-5 2e-5 4e-5; do   # trained at 4e-6: recall +0.1054, loss 2.6777
       run "recall_pg19_k1024_meta40_lr$lr" scripts/recall_probe.py --arm C --lora-rank 64 $MODEL $PROBE --inner-lr "$lr" --load-slow "$META"
       run "C_32k_k1024_t4_s40_evallr$lr" -m ttt.run --arm C --lora-rank 64 $MODEL --eval-sequences 32 --inner-lr "$lr" --load-slow "$META"
     done
     run recall_pg19_k1024_plain40_lr2e-5 scripts/recall_probe.py --arm C --lora-rank 64 $MODEL $PROBE --inner-lr 2e-5 --load-slow "$PLAIN" ;;
  *) echo "unknown part '$PART'"; exit 2 ;;
esac
echo "=== recall_lr part $PART: failed=$failed ==="
[ "$failed" -eq 0 ]
