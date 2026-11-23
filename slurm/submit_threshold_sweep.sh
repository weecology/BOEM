#!/bin/bash
#SBATCH --job-name=BOEM_thresh
#SBATCH --mail-type=END
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120GB
#SBATCH --time=04:00:00
#SBATCH --output=/home/b.weinstein/logs/BOEM%j.out
#SBATCH --error=/home/b.weinstein/logs/BOEM%j.err
#SBATCH --partition=hpg-b200
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1

# Re-derive predict.min_score per checkpoint on the pinned zero-shot holdout.
# Two checkpoints in one job so the curves are directly comparable:
#   55d29b2c/epoch08  job 38834235, best val_classification 0.01369  <- candidate
#   a09c6933/epoch16  job 36523583 (TIMEOUT), currently in boem_config  <- incumbent
# batch_size=1 and workers=5 are the optimum measured by worker sweep 39272218.

cd /blue/ewhite/b.weinstein/src/BOEM
CKPT_DIR=/blue/ewhite/b.weinstein/BOEM/training/checkpoints

uv run python scripts/threshold_sweep.py \
    --checkpoint "${CKPT_DIR}/55d29b2c7c5d42ff89c7fb698fe34255/epoch08-val_cls0.0137.ckpt" \
    --label 55d29b2c_e08 \
    --queue-sample 2000 --batch-size 1 --workers 5

echo
echo "############################################################"
echo

uv run python scripts/threshold_sweep.py \
    --checkpoint "${CKPT_DIR}/a09c69331af8496380cbf99e3859d656/epoch16-val_cls0.0163.ckpt" \
    --label a09c6933_e16 \
    --queue-sample 2000 --batch-size 1 --workers 5
