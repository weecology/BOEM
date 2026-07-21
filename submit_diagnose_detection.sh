#!/bin/bash
#SBATCH --job-name=BOEM_det_diag
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60GB
#SBATCH --time=02:00:00
#SBATCH --output=/home/b.weinstein/logs/det_diag_%j.out
#SBATCH --error=/home/b.weinstein/logs/det_diag_%j.err
#SBATCH --partition=hpg-b200
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1

ulimit -c 0

CKPT=/blue/ewhite/b.weinstein/BOEM/training/checkpoints/a09c69331af8496380cbf99e3859d656/epoch16-val_cls0.0163.ckpt
IMGDIR=/blue/ewhite/b.weinstein/BOEM/screened_images/JPG_20230426_110600
LIST=/blue/ewhite/b.weinstein/BOEM/detection_diag/seal_imgs.txt
OUT=/blue/ewhite/b.weinstein/BOEM/detection_diag
SCRIPT=/blue/ewhite/b.weinstein/src/BOEM/scripts/diagnose_detection.py

# A/B: same epoch16 checkpoint + same images, loaded under two DeepForest branches.
# balanced       = shared .venv (tmp/hpc-balanced-empty-frames) -- the branch the model was TRAINED on
# friendlybeaver = .venv-classification (claude/friendly-beaver) -- the branch the seals run used
echo "===== A: balanced-empty-frames branch (training branch) ====="
UV_PROJECT_ENVIRONMENT=/blue/ewhite/b.weinstein/src/BOEM/.venv \
  uv run --no-sync python -u "$SCRIPT" \
  --checkpoint "$CKPT" --image-list "$LIST" --image-dir "$IMGDIR" \
  --out-dir "$OUT" --tag balanced --min-score 0.01

echo "===== B: claude/friendly-beaver branch (what the seals run used) ====="
UV_PROJECT_ENVIRONMENT=/blue/ewhite/b.weinstein/src/BOEM/.venv-classification \
  uv run --no-sync python -u "$SCRIPT" \
  --checkpoint "$CKPT" --image-list "$LIST" --image-dir "$IMGDIR" \
  --out-dir "$OUT" --tag friendlybeaver --min-score 0.01
