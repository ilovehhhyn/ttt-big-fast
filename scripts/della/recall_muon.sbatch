#!/bin/bash
#SBATCH --account=arora
#SBATCH --job-name=recall_muon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/ARORA/hh9077/logs/%x-%j.out
# Does an orthogonalised update (Muon: the gradient's singular values are all set to 1) store more
# than normalized SGD at the same cost in loss? Un-tuned model, PG-19, 32K tokens, window 1024.
#   sbatch scripts/della/recall_muon.sbatch <inner-lr> [checkpoint]
# Without a checkpoint the un-tuned model is scored. With one (a result name, for example
# C_32k_k1024_t4_s40) its trained slow weights are loaded and only the inner rule changes.
# Measured on an A100 (jobs 14244957, 14244958): 20 s per sequence pass with Muon, so 32 pairs
# take 21 minutes and the 32-sequence loss 11. Those two jobs scored 16 of each; every result
# file records its own count.
set -uo pipefail
LR=${1:?usage: recall_muon.sbatch <inner-lr> [checkpoint]}
CKPT=${2:-}
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast; RES=/scratch/gpfs/ARORA/hh9077/results; DATA=/scratch/gpfs/ARORA/hh9077/data/pg19_32k
export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$ROOT"
cd "$ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
failed=0
if [ -z "$CKPT" ]; then
  ARM="--arm B"; RECALL_NAME="recall_pg19_k1024_untuned_muon_lr$LR"; LOSS_NAME="B_32k_k1024_muon_lr$LR"
else
  [ -e "$RES/$CKPT.ckpt" ] || { echo "checkpoint $RES/$CKPT.ckpt does not exist; pass the name of a finished training result"; exit 2; }
  ARM="--arm C --lora-rank 64 --load-slow $RES/$CKPT.ckpt"
  RECALL_NAME="recall_pg19_k1024_${CKPT}_muon_lr$LR"; LOSS_NAME="${CKPT}_muon_evallr$LR"
fi
MODEL="$ARM --mode eval --data $DATA --seq-len 32768 --chunk 1024 --window 1024 --prefix-segment 1024 --fast-blocks 4
       --remat-group 1 --remat-blocks --dtype bf16 --inner muon --inner-lr $LR"
run() {   # run <result name> <script and its arguments...>
  local name=$1; shift
  [ -e "$RES/$name.json" ] && { echo "[skip] $name"; return; }
  echo "[start] $name $(date +%H:%M:%S)"
  .venv/bin/python -u "$@" --out "$RES/$name.json"; local rc=$?
  [ $rc -eq 0 ] || { echo "[FAIL]  $name exit=$rc"; failed=$((failed + 1)); }
  echo "[end]   $name $(date +%H:%M:%S)"
}
run "$RECALL_NAME" scripts/recall_probe.py $MODEL --source-start 2048 --length 1024 --cue 32 --gap 17408 --pairs 32
run "$LOSS_NAME" -m ttt.run $MODEL --eval-sequences 32
echo "=== recall_muon lr=$LR: failed=$failed ==="
[ "$failed" -eq 0 ]
