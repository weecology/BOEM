#!/bin/bash
#SBATCH --job-name=BOEM_USGS   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI ran
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB
#SBATCH --time=48:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/detection_BOEM%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/detection_BOEM%j.err
#SBATCH --partition=hpg-b200
#SBATCH --ntasks-per-node=3
#SBATCH --gpus=3

uv run python scripts/prepare_USGS.py
uv run python scripts/USGS_backbone.py --batch_size 12 --workers 4 --max-empty-fraction 0.5

