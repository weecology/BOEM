#!/bin/bash
#SBATCH --job-name=BOEM_dino_size
#SBATCH --account=ewhite
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --time=10:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --output=/home/b.weinstein/logs/dino_size_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/dino_size_%A_%a.err
# DINOv3 RetinaNet (6fc49164 epoch14, trained at 1000) predicted at other input sizes, with the dense-flock chat's
# input_size_sweep.py (37732b5): validation strata, zero-shot sweep, July review load, cost. 800 = production 800/1333.
# Run from this worktree so its DeepForest PR #1473 venv and detection.load are used.
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/BOEM/.claude/worktrees/detector-architecture-comparison
export HF_HOME=/blue/ewhite/b.weinstein/.cache/huggingface
SIZES=(800 1200)
CKPT=/blue/ewhite/b.weinstein/BOEM/training/checkpoints/6fc49164bfa548d984f3af18f8352f2a/epoch14-val_cls0.0847.ckpt
.venv/bin/python -u scripts/input_size_sweep.py task ${SIZES[$SLURM_ARRAY_TASK_ID]} $CKPT dinov3_e14
