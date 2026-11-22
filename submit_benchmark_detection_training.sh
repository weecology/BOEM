#!/bin/bash
#SBATCH --job-name=BOEM_det_bench
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160GB
#SBATCH --time=02:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --output=/home/b.weinstein/logs/det_bench_%j.out
#SBATCH --error=/home/b.weinstein/logs/det_bench_%j.err
# Data loading vs GPU compute for USGS_backbone.py training (see script docstring).
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/BOEM
uv run python -u scripts/benchmark_detection_training.py --workers 8 16 24 32
