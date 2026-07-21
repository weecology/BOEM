#!/bin/bash
#SBATCH --job-name=BOEM_full_flight
#SBATCH --mail-type=END
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=90GB
#SBATCH --time=24:00:00
#SBATCH --output=/home/b.weinstein/logs/BOEM_full_flight%j.out
#SBATCH --error=/home/b.weinstein/logs/BOEM_full_flight%j.err
#SBATCH --partition=hpg-b200
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1

ulimit -c 0

# Usage: sbatch submit_upload_full_flight.sh JPG_20241107_135800 [--skip-annotated]
: ${1:?Usage: sbatch submit_upload_full_flight.sh FLIGHT_NAME [extra args]}

uv run python scripts/upload_full_flight.py "$@"
