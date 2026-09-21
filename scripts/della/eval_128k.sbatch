#!/bin/bash
#SBATCH --account=arora
#SBATCH --job-name=eval128k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/ARORA/hh9077/logs/%x-%j.out
# Longer context: PG-19 at T = 131072 (books of at least 131,073 tokens, so no sequence spans
# two books), nothing trained. Three questions, evaluation only:
#   1. what is context beyond an 8192-token window worth to a healthy model at 128K?
#   2. how broken is the un-tuned windowed model at 128K (arm A)?
#   3. what does TTT alone recover (arm B, normalized SGD 4e-6)?
# Too slow for the login node's ~10-minute limit, short enough for the test queue.
set -uo pipefail
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast; RES=/scratch/gpfs/ARORA/hh9077/results; DATA=/scratch/gpfs/ARORA/hh9077/data/pg19_128k
export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$ROOT"
cd "$ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -e "$DATA/val.json" ] || { echo "missing $DATA/val.json: prepare pg19_128k first"; exit 1; }
COMMON="--mode eval --data $DATA --seq-len 131072 --chunk 1024 --window 8192 --prefix-segment 8192
        --fast-blocks 4 --remat-group 1 --remat-blocks --dtype bf16 --eval-sequences 16"
[ -e "$RES/A_128k.json" ] || .venv/bin/python -u -m ttt.run $COMMON --arm A --out "$RES/A_128k.json"
[ -e "$RES/B_128k.json" ] || .venv/bin/python -u -m ttt.run $COMMON --arm B --inner normalized_sgd --inner-lr 4e-6 --out "$RES/B_128k.json"
# Question 1 is scripts/della/cv_128k.sbatch: one full-attention float32 forward at 128K takes
# about 6 minutes on an A100, so 12 sequences do not fit in this job (job 14238061 was
# cancelled at 46 minutes with 5 of 12 scored and nothing saved).
echo "=== eval_128k finished ==="
