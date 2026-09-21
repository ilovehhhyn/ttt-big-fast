#!/bin/bash
#SBATCH --account=arora
#SBATCH --job-name=prep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/gpfs/ARORA/hh9077/logs/%x-%j.out
# Tokenise a corpus on a CPU compute node, offline, from parquet shards downloaded beforehand.
# (The login node kills tokenisation after ~10 minutes of CPU; compute nodes have no internet.)
#   sbatch scripts/della/prep_cpu.sbatch <ttt.data.prepare args, including --data-files>
set -euo pipefail
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast
export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf HF_HUB_OFFLINE=1 PYTHONPATH="$ROOT" TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
cd "$ROOT"
.venv/bin/python -m ttt.data.prepare --tokenizer-id unsloth/Llama-3.2-1B "$@"
