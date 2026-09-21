#!/bin/bash
#SBATCH --account=arora
#SBATCH --job-name=cv128k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/ARORA/hh9077/logs/%x-%j.out
# What is context beyond an 8192-token window worth to a healthy model at T = 131072 (PG-19,
# books of at least 131,073 tokens)? One full-attention float32 forward at 128K takes about
# 6 minutes on an A100 (measured in job 14238061), so the sample is scored in disjoint pieces
# that each fit the test queue:
#   sbatch scripts/della/cv_128k.sbatch <skip> <count>      # count <= 7
# Piece <skip> writes results/context_value_128k_s8192_skip<skip>.json.
set -uo pipefail
SKIP=${1:?usage: cv_128k.sbatch <skip> <count>}; COUNT=${2:?usage: cv_128k.sbatch <skip> <count>}
[ "$COUNT" -le 7 ] || { echo "count $COUNT would not finish within the hour (about 6 minutes per sequence)"; exit 2; }
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast; RES=/scratch/gpfs/ARORA/hh9077/results; DATA=/scratch/gpfs/ARORA/hh9077/data/pg19_128k
export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$ROOT"
cd "$ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
out="$RES/context_value_128k_s8192_skip$SKIP.json"
[ -e "$out" ] && { echo "[skip] $out exists"; exit 0; }
.venv/bin/python -u scripts/context_value.py --data "$DATA" --seq-len 131072 --segment 8192 \
    --eval-sequences "$COUNT" --skip-sequences "$SKIP" --out "$out"
echo "=== cv_128k skip=$SKIP finished ==="
