#!/bin/bash
#SBATCH --account=arora
#SBATCH --job-name=memprobe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/gpfs/ARORA/hh9077/logs/%x-%j.out
set -euo pipefail
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast
export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf HF_HUB_OFFLINE=1 PYTHONPATH="$ROOT"
cd "$ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun --chdir="$ROOT" .venv/bin/python scripts/memory_probe.py "$@"
