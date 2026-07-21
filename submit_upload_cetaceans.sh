#!/bin/bash
#SBATCH --job-name=BOEM_upload_cetaceans
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=06:00:00
#SBATCH --output=/home/b.weinstein/logs/upload_cetaceans_%j.out
#SBATCH --error=/home/b.weinstein/logs/upload_cetaceans_%j.err
#SBATCH --partition=hpg-default
#SBATCH --ntasks-per-node=1

ulimit -c 0

# Upload every cached NEAQ frame containing a cetacean (~2,100 frames, whale + belly
# cameras) to the Label Studio review project so annotators can see the whales/dolphins.
# Reuses .prediction_cache/pool_predictions.csv -- runs NO models. Only cetacean boxes are
# attached as preannotations, keeping each import chunk well under the nginx body limit
# that the dense seal task hit. Runs on a compute node: the deepforest/torch import chain
# is slow on the login node, and the ~25GB SFTP image transfer should not run there.
# Set DRY_RUN=1 (sbatch --export=ALL,DRY_RUN=1) to report without touching Label Studio.
export UV_PROJECT_ENVIRONMENT=/blue/ewhite/b.weinstein/src/BOEM/.venv-classification

srun uv run --no-sync python -u \
    /blue/ewhite/b.weinstein/src/BOEM/scripts/upload_cetaceans_to_review.py \
    ${DRY_RUN:+--dry-run} ${EXTRA_ARGS:-}
