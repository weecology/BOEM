#!/bin/bash
#SBATCH --job-name=BOEM_prepare
#SBATCH --mail-type=END
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --time=08:00:00
#SBATCH --output=/home/b.weinstein/logs/prepare_USGS_%j.out
#SBATCH --error=/home/b.weinstein/logs/prepare_USGS_%j.err
#SBATCH --partition=hpg-default
#SBATCH --ntasks-per-node=1

ulimit -c 0

srun uv run python scripts/prepare_USGS.py
