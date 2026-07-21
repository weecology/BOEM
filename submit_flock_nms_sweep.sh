#!/bin/bash
#SBATCH --job-name=BOEM_flock_nms
#SBATCH --mail-type=END
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --output=/home/b.weinstein/logs/flock_nms_%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/flock_nms_%x_%j.err
#SBATCH --partition=hpg-b200
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1

ulimit -c 0

CKPT="${CKPT:-/blue/ewhite/b.weinstein/BOEM/training/checkpoints/a09c69331af8496380cbf99e3859d656/epoch16-val_cls0.0163.ckpt}"
OUT="${OUT:-/blue/ewhite/b.weinstein/BOEM/flock_nms_sweep/${SLURM_JOB_ID}}"

echo "CKPT=$CKPT"
echo "OUT=$OUT"

srun uv run python scripts/sweep_flock_nms.py \
    --checkpoint "$CKPT" \
    --out-dir "$OUT" \
    --n-images 10 \
    --nms 0.05 0.15 0.3 0.5 0.7 \
    --score 0.1 0.3 0.85 \
    --patch 1000 500
