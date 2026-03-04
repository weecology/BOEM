#!/bin/bash
#
# Submit a single flight flythrough video job (no GPU).
#
# Usage:
#   sbatch submit_flythrough.sh /blue/ewhite/b.weinstein/BOEM/imagery/JPG_20241219_120500
#   sbatch submit_flythrough.sh  (uses default flight below)
#
# Output video: /blue/ewhite/b.weinstein/BOEM/flight_videos/<flight_name>_flythrough.avi
#

#SBATCH --job-name=flythrough
#SBATCH --mail-type=END
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB
#SBATCH --time=08:00:00
#SBATCH --output=/home/b.weinstein/logs/flythrough_%j.out
#SBATCH --error=/home/b.weinstein/logs/flythrough_%j.err
#SBATCH --ntasks-per-node=1

FLIGHT_DIR="${1:-/blue/ewhite/b.weinstein/BOEM/imagery/JPG_20260201_093500}"

cd /blue/ewhite/b.weinstein/src/BOEM || exit 1
uv run python scripts/flight_flythrough_video.py "$FLIGHT_DIR"
