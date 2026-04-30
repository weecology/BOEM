#!/bin/bash
#SBATCH --job-name=BOEM_USGS_balanced
#SBATCH --mail-type=END
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=50GB
#SBATCH --time=12:00:00
#SBATCH --output=/home/b.weinstein/logs/classification_BOEM_balanced_%j.out
#SBATCH --error=/home/b.weinstein/logs/classification_BOEM_balanced_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

ulimit -c 0

# prepare_USGS.py already run by the unbalanced job; skip crop regeneration here.
uv run python -u scripts/prepare_USGS.py --no-generate-detection-crops --no-update-labels

export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
uv run python -u /blue/ewhite/b.weinstein/src/BOEM/scripts/USGS_classification.py classification_model.balance_classes=true
