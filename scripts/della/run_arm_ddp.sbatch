#!/bin/bash
#SBATCH --account=arora
#SBATCH --job-name=ttt-arm-ddp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
# 80 GiB cards only (see run_arm.sbatch): 32K training peaks at 68 GiB per GPU.
#SBATCH --constraint=gpu80
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gpfs/ARORA/hh9077/logs/%x-%j.out
# Data-parallel training: one process per GPU, sequences sharded across them
# (ttt/train/distributed.py). The sequences of an outer step are independent, so R GPUs cut
# the wall-clock of a step by R and the result equals the single-GPU run up to float rounding.
# --tokens-per-step must give a number of sequences divisible by the number of GPUs.
# Override the GPU count with:  sbatch --ntasks-per-node=N --gres=gpu:N ...
set -euo pipefail
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast
export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ttt.run reads SLURM_PROCID / SLURM_NTASKS / SLURM_LOCALID itself; it needs only a rendezvous.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
cd "$ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun .venv/bin/python -m ttt.run "$@"
