#!/bin/bash
#SBATCH --job-name=BOEM_cls_balance
#SBATCH --mail-type=END
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --time=24:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --output=/home/b.weinstein/logs/cls_balance_%j.out
#SBATCH --error=/home/b.weinstein/logs/cls_balance_%j.err

# Class-balance cap bracket against 0139af8c (job 52121225). Same code (branch
# flat-drop-genus-sp @ cdc49fe for every file the classifier reads), same unchanged
# crop pool, same split seed, so the val split must come out identical to 0139af8c's.
# Usage: sbatch submit_class_balance_bracket.sh <factor>   (4.0 = noise-control rerun, null = no cap)
FACTOR=${1:?usage: sbatch submit_class_balance_bracket.sh <factor|null>}
ulimit -c 0
cd /blue/ewhite/b.weinstein/src/BOEM/.claude/worktrees/class-balance-bracket

# Same venv as 52121225; --no-sync so this job cannot mutate it for other sessions.
export UV_PROJECT_ENVIRONMENT=/blue/ewhite/b.weinstein/src/BOEM/.venv-classification
uv run --no-sync python -u scripts/USGS_classification.py \
    classification_model.use_metadata=False \
    classification_model.class_balance_factor=$FACTOR
