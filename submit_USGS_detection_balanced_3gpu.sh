#!/bin/bash
#SBATCH --job-name=BOEM_det_bal_3gpu
#SBATCH --mail-type=END
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240GB
#SBATCH --time=48:00:00
#SBATCH --output=/home/b.weinstein/logs/detection_BOEM_balanced_3gpu_%j.out
#SBATCH --error=/home/b.weinstein/logs/detection_BOEM_balanced_3gpu_%j.err
#SBATCH --partition=hpg-b200
#SBATCH --gpus=3
# One srun task per GPU so PyTorch-Lightning DDP (SLURMEnvironment) maps each rank to a GPU.
#SBATCH --ntasks-per-node=3

ulimit -c 0

export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO

# --no-sync: use the existing shared .venv as-is (already on the detection branch
# tmp/hpc-balanced-empty-frames). pyproject is temporarily pointed at the classification
# branch, and the single-GPU detection job 36523583 is training off this same .venv, so we
# must NOT re-sync/mutate it here.
srun uv run --no-sync python scripts/USGS_backbone.py \
    --batch_size 64 \
    --workers 8 \
    --lr 0.0001 \
    --epochs 20 \
    --positive-batch-fraction 0.5
