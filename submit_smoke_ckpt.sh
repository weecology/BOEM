#!/bin/bash
#SBATCH --job-name=BOEM_smoke_ckpt
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --output=/home/b.weinstein/logs/smoke_ckpt_%j.out
#SBATCH --error=/home/b.weinstein/logs/smoke_ckpt_%j.err
#SBATCH --partition=hpg-default
#SBATCH --ntasks-per-node=1

ulimit -c 0

# Can the classification-branch DeepForest (claude/friendly-beaver, PR #1334) load the
# detection checkpoint that was trained on the balanced-empty-frames branch? If not, the
# seals pipeline can't run both models in one process.
export UV_PROJECT_ENVIRONMENT=/blue/ewhite/b.weinstein/src/BOEM/.venv-classification
srun uv run --no-sync python -u /blue/ewhite/b.weinstein/src/BOEM/scripts/smoke_ckpt.py
