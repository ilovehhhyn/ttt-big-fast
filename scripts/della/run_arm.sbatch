#!/bin/bash
#SBATCH --account=arora
#SBATCH --job-name=ttt-arm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gpfs/ARORA/hh9077/logs/%x-%j.out
set -euo pipefail
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast
export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf
export HF_HUB_OFFLINE=1          # compute nodes have no internet; use the cached snapshot
export TOKENIZERS_PARALLELISM=false
# The second-order path allocates and frees large attention-score buffers; without
# expandable segments the allocator fragments and OOMs ~2 GiB below the real limit.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun .venv/bin/python -m ttt.run "$@"
