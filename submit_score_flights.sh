#!/bin/bash
# Score held-out flights for the land filter. CPU-bound (~440 ms/frame), so this wants
# real cores -- an interactive shell gets one and the process pool does nothing.

#SBATCH --job-name=score_land
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --output=/home/b.weinstein/logs/score_land_%j.out
#SBATCH --error=/home/b.weinstein/logs/score_land_%j.err

cd "${SLURM_SUBMIT_DIR}"
uv run python scripts/score_flights.py --n 5000
