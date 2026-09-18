#!/bin/bash
#SBATCH --account=arora --job-name=scanlr -N1 --ntasks=1 -c8 --mem=160G --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/gpfs/ARORA/hh9077/logs/%x-%j.out
# Arm B (TTT-naive: inner loop on, nothing meta-learned) across inner step sizes.
# The reference point is e2e's own rule: clip_by_global_norm(1)+sgd(1) gives a
# per-element RMS step of 1/sqrt(n_fast) = 7.05e-5 for our 201M-parameter fast set.
set -euo pipefail
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast
export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf HF_HUB_OFFLINE=1 PYTHONPATH="$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
cd "$ROOT"
nvidia-smi --query-gpu=name --format=csv,noheader
COMMON="--arm B --mode eval --data /scratch/gpfs/ARORA/hh9077/data/dclm8k --fast-blocks 4 --remat-group 1 --remat-blocks --eval-sequences 32"
for lr in 0 7e-6 2e-5 7e-5 2e-4 7e-4; do
  echo "########## inner_lr_rms=$lr ##########"
  srun --chdir="$ROOT" .venv/bin/python -m ttt.run $COMMON --inner normalized_sgd --inner-lr "$lr" \
       --out "results/B_lr${lr}.json" || true
done
echo "########## e2e exact rule: clipped_sgd lr=1 tau=1 ##########"
srun --chdir="$ROOT" .venv/bin/python -m ttt.run $COMMON --inner clipped_sgd --inner-lr 1.0 --out results/B_clipped.json || true
