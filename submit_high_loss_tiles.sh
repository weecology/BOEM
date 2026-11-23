#!/bin/bash
#SBATCH --job-name=BOEM_high_loss
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --time=06:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --output=/home/b.weinstein/logs/high_loss_%j.out
#SBATCH --error=/home/b.weinstein/logs/high_loss_%j.err
# Per-tile training loss on a detector's train + val tiles (upload step runs separately on CPU):
#   sbatch submit_high_loss_tiles.sh <checkpoint.ckpt> <train_csv> <val_csv>
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/BOEM
uv run python -u scripts/find_high_loss_tiles.py score "$@"
