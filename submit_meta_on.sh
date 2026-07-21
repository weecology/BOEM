#!/bin/bash
#SBATCH --job-name=meta_on   # A/B run: classification WITH flight metadata
#SBATCH --mail-type=END
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=50GB
#SBATCH --time=12:00:00
#SBATCH --output=/home/b.weinstein/logs/meta_on_%j.out
#SBATCH --error=/home/b.weinstein/logs/meta_on_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

ulimit -c 0

export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO

export UV_PROJECT_ENVIRONMENT=/blue/ewhite/b.weinstein/src/BOEM/.venv-classification
uv run python -u /blue/ewhite/b.weinstein/src/BOEM/scripts/USGS_classification.py \
    classification_model.use_metadata=true
