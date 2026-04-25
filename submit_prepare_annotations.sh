#!/bin/bash
# One SLURM job: request many CPUs; prepare_USGS.py parallelizes split_raster per image.
# Run from repo root on the cluster: sbatch submit_prepare_annotations.sh

#SBATCH --job-name=prep_ann
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --output=/home/b.weinstein/logs/prep_ann_%j.out
#SBATCH --error=/home/b.weinstein/logs/prep_ann_%j.err

cd "${SLURM_SUBMIT_DIR}"
uv run python scripts/prepare_USGS.py
