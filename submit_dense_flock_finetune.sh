#!/bin/bash
#SBATCH --job-name=BOEM_dense_p1
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --time=08:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --array=0-9%4
#SBATCH --output=/home/b.weinstein/logs/dense_flock_p1_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/dense_flock_p1_%A_%a.err
# Dense-flock Phase 1: fine-tune c6ecec12 under different training distributions, same steps.
# Task 0 = no fine-tune reference; 1-9 = {random, dense, weighted} x seeds {0, 1, 2}.
# Outputs: /blue/ewhite/b.weinstein/BOEM/dense_flock/phase1/
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/BOEM
ARMS=(none random random random dense dense dense weighted weighted weighted)
SEEDS=(0 0 1 2 0 1 2 0 1 2)
i=${SLURM_ARRAY_TASK_ID}
uv run python -u scripts/dense_flock_finetune.py "${ARMS[$i]}" "${SEEDS[$i]}"
