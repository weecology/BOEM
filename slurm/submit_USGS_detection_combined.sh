#!/bin/bash
#SBATCH --job-name=BOEM_det_combined
#SBATCH --mail-type=END
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --time=24:00:00
#SBATCH --output=/home/b.weinstein/logs/detection_BOEM%j.out
#SBATCH --error=/home/b.weinstein/logs/detection_BOEM%j.err
#SBATCH --partition=hpg-b200
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1

ulimit -c 0

export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO

srun uv run python scripts/USGS_backbone.py \
    --batch_size 64 \
    --workers 8 \
    --lr 0.001 \
    --max-empty-fraction 0.1 \
    --max-test-empty-fraction 0.25 \
    --early-stopping-patience 5
