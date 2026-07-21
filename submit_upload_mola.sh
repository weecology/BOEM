#!/bin/bash
#SBATCH --job-name=BOEM_upload_mola
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=/home/b.weinstein/logs/upload_mola_%j.out
#SBATCH --error=/home/b.weinstein/logs/upload_mola_%j.err
#SBATCH --partition=hpg-default
#SBATCH --ntasks-per-node=1

ulimit -c 0

# Upload a flight-stratified sample (<=5 frames/flight, ~157 frames) of the suspected
# "Mola mola" false-positive predictions from the NEAQ caches to the Label Studio review
# project, so annotators can characterize/correct the overprediction. Runs NO models.
# Set DRY_RUN=1 (sbatch --export=ALL,DRY_RUN=1) to report without touching Label Studio.
export UV_PROJECT_ENVIRONMENT=/blue/ewhite/b.weinstein/src/BOEM/.venv-classification

srun uv run --no-sync python -u \
    /blue/ewhite/b.weinstein/src/BOEM/scripts/upload_mola_sample_to_review.py \
    ${DRY_RUN:+--dry-run}
