#!/bin/bash
#SBATCH --job-name=cls_confusion
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=50GB
#SBATCH --time=00:30:00
#SBATCH --output=/home/b.weinstein/logs/cls_confusion_%j.out
#SBATCH --error=/home/b.weinstein/logs/cls_confusion_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

ulimit -c 0

# Val-split confusion matrix for one classifier checkpoint. Pass the comet id:
#   sbatch submit_classifier_confusion.sh a3dc30a085f5442393736ecd96b564c5
export UV_PROJECT_ENVIRONMENT=/blue/ewhite/b.weinstein/src/BOEM/.venv-classification
cd /blue/ewhite/b.weinstein/src/BOEM
srun uv run --no-sync python -u scripts/classifier_confusion.py "$1"
