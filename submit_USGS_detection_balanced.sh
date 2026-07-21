#!/bin/bash
#SBATCH --job-name=BOEM_det_balanced
#SBATCH --mail-type=END
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120GB
#SBATCH --time=24:00:00
#SBATCH --output=/home/b.weinstein/logs/detection_BOEM_balanced_%j.out
#SBATCH --error=/home/b.weinstein/logs/detection_BOEM_balanced_%j.err
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
    --lr 0.0001 \
    --epochs 20 \
    --positive-batch-fraction 0.5
