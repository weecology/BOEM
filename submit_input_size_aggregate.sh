#!/bin/bash
#SBATCH --job-name=BOEM_size_agg
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=03:00:00
#SBATCH --partition=hpg-default
#SBATCH --output=/home/b.weinstein/logs/size_sweep_aggregate_%j.out
#SBATCH --error=/home/b.weinstein/logs/size_sweep_aggregate_%j.err
# CPU aggregate of the c6ecec12 input-size sweep (52694865): tables, paired zero-shot bootstrap, foam sheets, box sizes.
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/BOEM/.claude/worktrees/dense-flock
uv run python -u scripts/input_size_sweep.py aggregate
