#!/bin/bash
#SBATCH --job-name=BOEM_dense_insize
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --time=06:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --output=/home/b.weinstein/logs/dense_flock_insize_%j.out
#SBATCH --error=/home/b.weinstein/logs/dense_flock_insize_%j.err
# c6ecec12 at inference input 800/1000/1200: July reviewer load (zero-shot is boem-d5 job 52502538).
# Runs from the dense-flock worktree. Outputs: /blue/ewhite/b.weinstein/BOEM/dense_flock/input_size/
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/BOEM/.claude/worktrees/dense-flock
uv run python -u scripts/dense_flock_input_size.py "$@"
