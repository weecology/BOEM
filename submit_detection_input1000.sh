#!/bin/bash
#SBATCH --job-name=BOEM_det_1000
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
# c6ecec12's recipe (submit_USGS_detection_combined.sh: deepforest-bird init, batch 64, lr 0.001,
# ReduceLROnPlateau, early stop 5, 30 epochs) with ONE change: tiles enter the network at native
# 1000 px instead of torchvision's 800 px resize, in training and evaluation alike.
# Same split files c6ecec12 trained/validated on, read as-is (no resampling).
# Production venv (DeepForest friendly-beaver), not this worktree's DINOv3 venv, so the DeepForest
# version is not a second variable.
ulimit -c 0
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0

CROPS=/blue/ewhite/b.weinstein/BOEM/training/crops
cd /blue/ewhite/b.weinstein/src/BOEM/.claude/worktrees/detector-architecture-comparison
srun /blue/ewhite/b.weinstein/src/BOEM/.venv/bin/python -u scripts/USGS_backbone.py \
    --batch_size 64 \
    --workers 16 \
    --lr 0.001 \
    --early-stopping-patience 5 \
    --input-size 1000 \
    --train-csv $CROPS/train_max_empty_0_10.csv \
    --val-csv $CROPS/test_max_empty_0_25.csv \
    --exp-name detection_input1000
