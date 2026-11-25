#!/bin/bash
#SBATCH --job-name=geo_eval
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24GB
#SBATCH --time=00:45:00
#SBATCH --output=/home/b.weinstein/logs/geo_eval_%j.out
#SBATCH --error=/home/b.weinstein/logs/geo_eval_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

ulimit -c 0
# Held-out labelled-box logits for the geo-prior experiments (E1/E2).
export UV_PROJECT_ENVIRONMENT=/blue/ewhite/b.weinstein/src/BOEM/.venv-classification
cd /blue/ewhite/b.weinstein/src/BOEM/.claude/worktrees/species-geo-priors
srun uv run --no-sync python -u -m scripts.geo_priors_eval_logits "$@"
