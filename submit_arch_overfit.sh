#!/bin/bash
#SBATCH --job-name=BOEM_arch_overfit
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --time=03:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --output=/home/b.weinstein/logs/arch_overfit_%j.out
#SBATCH --error=/home/b.weinstein/logs/arch_overfit_%j.err
# Phase 0 learning check: each architecture trains AND validates on the same 256 bird tiles
# (overfit_256.csv) for 30 epochs at native 1000 px. A model that is learning drives mAP/recall up on
# its own training tiles; one that stays near 0 is misconfigured, whatever the full run would say.
# Phase 1 recipes (per-GPU batch 16), same as the planned 2xB200 runs except for epochs.
set -uo pipefail
cd /blue/ewhite/b.weinstein/src/BOEM/.claude/worktrees/detector-architecture-comparison
export HF_HOME=/blue/ewhite/b.weinstein/.cache/huggingface
CSV=/blue/ewhite/b.weinstein/BOEM/architecture_comparison/overfit/overfit_256.csv
COMMON=(--train-csv $CSV --val-csv $CSV --zero-shot-csv $CSV --input-size 1000 --workers 16 --epochs 30 --n-log-images 2
        --checkpoint-root /blue/ewhite/b.weinstein/BOEM/architecture_comparison/overfit/checkpoints --save-top-k 1)
run() { echo; echo "################ $1 $(date +%T) ################"; shift
        .venv/bin/python -u scripts/USGS_backbone.py "${COMMON[@]}" "$@"; echo "exit $? $(date +%T)"; }
run retinanet --architecture retinanet --batch_size 16 --lr 0.001 --exp-name arch_overfit_retinanet
run retinanet_dinov3 --architecture retinanet_dinov3 --batch_size 16 --lr 0.01 --cosine \
    --precision bf16-mixed --gradient-clip 1.0 --exp-name arch_overfit_dinov3
run DeformableDetr --architecture DeformableDetr --batch_size 16 --optimizer AdamW --lr 0.0001 \
    --weight-decay 0.0001 --cosine --gradient-clip 0.1 --monitor val_loss --exp-name arch_overfit_detr
