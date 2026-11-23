#!/bin/bash
#SBATCH --job-name=BOEM_arch_smoke
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --time=03:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --output=/home/b.weinstein/logs/arch_smoke_%j.out
#SBATCH --error=/home/b.weinstein/logs/arch_smoke_%j.err
# Phase 0 plumbing: each architecture builds, trains 1 epoch on SMOKE random train tiles, validates,
# runs the pre/post zero-shot validate and saves a checkpoint, all at native 1000 px input, in this
# worktree's DeepForest PR #1473 venv. Lightning's it/s line gives first throughput numbers.
set -uo pipefail
cd /blue/ewhite/b.weinstein/src/BOEM/.claude/worktrees/detector-architecture-comparison
export HF_HOME=/blue/ewhite/b.weinstein/.cache/huggingface
export NCCL_IB_DISABLE=1 NCCL_NVLS_ENABLE=0
SMOKE=${SMOKE:-512}
COMMON=(--smoke "$SMOKE" --input-size 1000 --workers 16)
nvidia-smi -L
# Peak GPU memory per arm comes from this trace (timestamped, every 5 s).
nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 5 > /home/b.weinstein/logs/arch_smoke_${SLURM_JOB_ID}_gpumem.csv &

run() { echo; echo "################ $1 $(date +%T) ################"; shift
        .venv/bin/python -u scripts/USGS_backbone.py "${COMMON[@]}" "$@"; echo "exit $? $(date +%T)"; }

# R50 reference, production recipe, in the new DeepForest (version-drift check of the training path).
run retinanet --architecture retinanet --batch_size 64 --lr 0.001 --exp-name arch_smoke_retinanet
# DINOv3 ViT-L/16 sat493m, unfrozen: MillionTrees E007 recipe, lr scaled linearly 0.04 @ 64 -> 0.01 @ 16.
run retinanet_dinov3 --architecture retinanet_dinov3 --batch_size 16 --lr 0.01 --cosine \
    --precision bf16-mixed --gradient-clip 1.0 --exp-name arch_smoke_dinov3
# Deformable DETR (COCO): standard AdamW 1e-4, wd 1e-4, clip 0.1.
run DeformableDetr --architecture DeformableDetr --batch_size 16 --optimizer AdamW --lr 0.0001 \
    --weight-decay 0.0001 --cosine --gradient-clip 0.1 --exp-name arch_smoke_detr
