#!/bin/bash
#SBATCH --job-name=BOEM_det_combined
#SBATCH --mail-type=END
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --time=48:00:00
#SBATCH --output=/home/b.weinstein/logs/detection_BOEM%j.out
#SBATCH --error=/home/b.weinstein/logs/detection_BOEM%j.err
#SBATCH --partition=hpg-b200
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1

ulimit -c 0
# 16 CPU / 16 workers (2026-11-22, job 52339591 benchmark): at 8 workers training was loader-bound
# (~108 img/s loader vs 197 img/s B200 step), ~48 min/epoch, and 52121224 hit the old 24h limit.
# ~15 img/s per worker, so 16 keeps one B200 fed: ~22 min/epoch, ~11h for 30 epochs.

export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO

srun uv run python scripts/USGS_backbone.py \
    --batch_size 64 \
    --workers 16 \
    --lr 0.001 \
    --max-empty-fraction 0.1 \
    --max-test-empty-fraction 0.25 \
    --early-stopping-patience 5
