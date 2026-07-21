#!/bin/bash
#SBATCH --job-name=BOEM_meta_check
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=00:20:00
#SBATCH --output=/home/b.weinstein/logs/meta_check_%j.out
#SBATCH --error=/home/b.weinstein/logs/meta_check_%j.err
#SBATCH --partition=hpg-b200
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1

ulimit -c 0
export UV_PROJECT_ENVIRONMENT=/blue/ewhite/b.weinstein/src/BOEM/.venv-classification
srun uv run --no-sync python scripts/verify_metadata_fallback.py
