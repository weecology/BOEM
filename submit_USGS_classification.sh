#!/bin/bash
#SBATCH --job-name=BOEM_USGS   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI ran
#SBATCH --cpus-per-task=5
#SBATCH --mem=50GB
#SBATCH --time=12:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/classification_BOEM%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/classification_BOEM%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

ulimit -c 0


uv run python scripts/prepare_USGS.py

export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
uv run python /blue/ewhite/b.weinstein/src/BOEM/scripts/USGS_classification.py