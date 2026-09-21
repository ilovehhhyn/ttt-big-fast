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
#   sbatch scripts/della/recall_muon.sbatch <inner-lr>
# `--inner muon` has never run on a GPU. Its Newton-Schulz iteration is 9e12 FLOPs per chunk in
# float32 (about 30 s per sequence on an A100, estimated, not measured), so this job scores the
# FIRST 16 of the usual 32 pairs and 16 sequences for the loss; the normalized-SGD results hold
# per-pair and per-sequence values, so the same 16 can be compared. Recall runs first.
set -uo pipefail
LR=${1:?usage: recall_muon.sbatch <inner-lr>}
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast; RES=/scratch/gpfs/ARORA/hh9077/results; DATA=/scratch/gpfs/ARORA/hh9077/data/pg19_32k
export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$ROOT"
cd "$ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
failed=0
MODEL="--arm B --mode eval --data $DATA --seq-len 32768 --chunk 1024 --window 1024 --prefix-segment 1024 --fast-blocks 4
       --remat-group 1 --remat-blocks --dtype bf16 --inner muon --inner-lr $LR"
run() {   # run <result name> <script and its arguments...>
  local name=$1; shift
  [ -e "$RES/$name.json" ] && { echo "[skip] $name"; return; }
  echo "[start] $name $(date +%H:%M:%S)"
  .venv/bin/python -u "$@" --out "$RES/$name.json"; local rc=$?
  [ $rc -eq 0 ] || { echo "[FAIL]  $name exit=$rc"; failed=$((failed + 1)); }
  echo "[end]   $name $(date +%H:%M:%S)"
}
run "recall_pg19_k1024_untuned_muon_lr$LR" scripts/recall_probe.py $MODEL --source-start 2048 --length 1024 --cue 32 --gap 17408 --pairs 16
run "B_32k_k1024_muon_lr$LR" -m ttt.run $MODEL --eval-sequences 16
echo "=== recall_muon lr=$LR: failed=$failed ==="
[ "$failed" -eq 0 ]
