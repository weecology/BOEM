#!/bin/bash
#SBATCH --job-name=BOEM_eval_pre
#SBATCH --mail-type=END
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH --time=01:00:00
#SBATCH --output=/home/b.weinstein/logs/eval_pre_cleanup_%j.out
#SBATCH --error=/home/b.weinstein/logs/eval_pre_cleanup_%j.err
#SBATCH --partition=hpg-b200
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1

ulimit -c 0

srun uv run python scripts/eval_pre_cleanup_checkpoint.py
