#!/bin/bash
#SBATCH --job-name=BOEM_reclass_old
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60GB
#SBATCH --time=06:00:00
#SBATCH --output=/home/b.weinstein/logs/reclass_old_%j.out
#SBATCH --error=/home/b.weinstein/logs/reclass_old_%j.err
#SBATCH --partition=hpg-b200
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1

ulimit -c 0

# Re-classify the cached July detection boxes with the April classifier (4c002d6b, no
# metadata). Detection is NOT re-run — boxes come from the .prediction_cache files — so this
# isolates the classifier and gives harbor-seal boxes to upload for review.
export UV_PROJECT_ENVIRONMENT=/blue/ewhite/b.weinstein/src/BOEM/.venv-classification

srun uv run --no-sync python -u /blue/ewhite/b.weinstein/src/BOEM/scripts/reclassify_with_old_model.py \
    --checkpoint /blue/ewhite/b.weinstein/BOEM/training/classification/checkpoints/buffer_30/4c002d6bfb654e10ab9ae99aa0451f81.ckpt \
    --out /blue/ewhite/b.weinstein/BOEM/detection_diag/reclassified_old_vs_new.csv
