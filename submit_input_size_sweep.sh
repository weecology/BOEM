#!/bin/bash
#SBATCH --job-name=BOEM_size_sweep
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --time=16:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --array=0-5%3
#SBATCH --output=/home/b.weinstein/logs/size_sweep_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/size_sweep_%A_%a.err
# c6ecec12/epoch27 inference input-size sweep, one size per task (val + zero-shot + July + cost).
# Aggregate afterwards on CPU: uv run python scripts/input_size_sweep.py aggregate
# Outputs: /blue/ewhite/b.weinstein/BOEM/dense_flock/size_sweep/in<size>/
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/BOEM/.claude/worktrees/dense-flock
SIZES=(800 1000 1200 1400 1600 2000)
uv run python -u scripts/input_size_sweep.py task "${SIZES[$SLURM_ARRAY_TASK_ID]}"
