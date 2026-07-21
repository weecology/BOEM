#!/bin/bash
#SBATCH --job-name=BOEM_upload_seals
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=/home/b.weinstein/logs/upload_seals_%j.out
#SBATCH --error=/home/b.weinstein/logs/upload_seals_%j.err
#SBATCH --partition=hpg-default
#SBATCH --ntasks-per-node=1

ulimit -c 0

# Upload the April classifier's harbor-seal candidates to the Label Studio review project.
# Runs on a compute node: the deepforest/torch import chain takes ~7+ min on the login node.
# Set DRY_RUN=1 (sbatch --export=ALL,DRY_RUN=1) to report what would upload without
# touching Label Studio.
export UV_PROJECT_ENVIRONMENT=/blue/ewhite/b.weinstein/src/BOEM/.venv-classification

srun uv run --no-sync python -u \
    /blue/ewhite/b.weinstein/src/BOEM/scripts/upload_old_harbor_seals_to_review.py \
    ${DRY_RUN:+--dry-run}
