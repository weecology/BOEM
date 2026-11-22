#!/bin/bash
#SBATCH --job-name=BOEM_review_load
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --time=12:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --output=/home/b.weinstein/logs/review_load_%j.out
#SBATCH --error=/home/b.weinstein/logs/review_load_%j.err
# Reviewer workload vs detection threshold on 5,000 random July frames (see script docstring).
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/BOEM
uv run python -u scripts/review_load_by_threshold.py \
    /blue/ewhite/b.weinstein/BOEM/training/checkpoints/1968e962113f4c57bc6bf5f07fae1b25/epoch29-val_cls0.0899.ckpt \
    /blue/ewhite/b.weinstein/BOEM/training/checkpoints/c6ecec12e078416c9fe645c89bc418cf/epoch27-val_cls0.0867.ckpt
