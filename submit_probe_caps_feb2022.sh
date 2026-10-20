#!/bin/bash
#SBATCH --job-name=BOEM_cap_feb2022
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --output=/home/b.weinstein/logs/cap_probe_feb2022_%j.out
#SBATCH --error=/home/b.weinstein/logs/cap_probe_feb2022_%j.err
#SBATCH --partition=hpg-b200
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1

ulimit -c 0

CKPT="/blue/ewhite/b.weinstein/BOEM/training/checkpoints/a09c69331af8496380cbf99e3859d656/epoch16-val_cls0.0163.ckpt"
FLOCK="/blue/ewhite/b.weinstein/BOEM/detection_cap_probe/feb2022"
OUT="/blue/ewhite/b.weinstein/BOEM/detection_cap_probe/feb2022_run"

echo "CKPT=$CKPT"; echo "FLOCK=$FLOCK"; echo "OUT=$OUT"

srun uv run python scripts/probe_detection_caps.py \
    --checkpoint "$CKPT" \
    --flock-dir "$FLOCK" \
    --out-dir "$OUT" \
    --n-images 12 \
    --patch 1000 \
    --score 0.05
