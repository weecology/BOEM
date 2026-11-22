#!/bin/bash
#SBATCH --job-name=BOEM_det_eval
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=03:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --output=/home/b.weinstein/logs/det_eval_%j.out
#SBATCH --error=/home/b.weinstein/logs/det_eval_%j.err
# Validation (in-distribution) box metrics for one detection checkpoint:
#   sbatch submit_eval_detection.sh <checkpoint.ckpt> [val_csv]
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/BOEM
uv run python -u scripts/eval_detection_checkpoint.py "$@"
