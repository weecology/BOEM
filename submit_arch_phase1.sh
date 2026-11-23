#!/bin/bash
#SBATCH --job-name=BOEM_arch_p1
#SBATCH --account=ewhite
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --cpus-per-task=12
#SBATCH --mem=240GB
#SBATCH --time=36:00:00
#SBATCH --partition=hpg-b200
#SBATCH --output=/home/b.weinstein/logs/arch_p1_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/arch_p1_%A_%a.err
# Architecture comparison Phase 1 (Ben 2026-11-23: 2 B200 per arm, 15 epochs).
#   0 = RetinaNet + DINOv3 ViT-L/16 sat493m, unfrozen (MillionTrees E007 recipe: SGD lr 0.02 @ batch 32, cosine, bf16, clip 1.0)
#   1 = Deformable DETR, COCO init (AdamW 1e-4, wd 1e-4, cosine, clip 0.1)
# Same data as production c6ecec12 (its own train_max_empty_0_10.csv / test_max_empty_0_25.csv, read as-is), same
# augmentations, native 1000 px input. Batch 16 per GPU x 2 (DDP). Fixed 15-epoch cosine budget, no early stopping;
# every epoch is checkpointed and the best is picked by the monitored validation loss within each run.
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/BOEM/.claude/worktrees/detector-architecture-comparison
export HF_HOME=/blue/ewhite/b.weinstein/.cache/huggingface
export NCCL_IB_DISABLE=1 NCCL_NVLS_ENABLE=0
CROPS=/blue/ewhite/b.weinstein/BOEM/training/crops
COMMON=(--train-csv $CROPS/train_max_empty_0_10.csv --val-csv $CROPS/test_max_empty_0_25.csv
        --input-size 1000 --batch_size 16 --workers 10 --epochs 15 --cosine)
case $SLURM_ARRAY_TASK_ID in
  0) ARM=(--architecture retinanet_dinov3 --lr 0.02 --precision bf16-mixed --gradient-clip 1.0
          --exp-name arch_p1_dinov3_sat493m) ;;
  1) ARM=(--architecture DeformableDetr --optimizer AdamW --lr 0.0001 --weight-decay 0.0001
          --gradient-clip 0.1 --monitor val_loss --exp-name arch_p1_deformable_detr) ;;
esac
nvidia-smi -L
# SMOKE=N (sbatch --export=ALL,SMOKE=512): same arms and DDP layout on N tiles per split, 3 epochs, scratch dir.
srun .venv/bin/python -u scripts/USGS_backbone.py "${COMMON[@]}" "${ARM[@]}" ${SMOKE:+--smoke $SMOKE}
