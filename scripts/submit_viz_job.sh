#!/bin/bash

#SBATCH --job-name=camera-b-viz
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=20:00
#SBATCH --partition=hpg-turin
#SBATCH --output=/blue/ewhite/b.weinstein/BOEM/NOAA/Camera\ B/slurm_viz_%j.log

# Load environment
cd /blue/ewhite/b.weinstein/src/BOEM

# Run visualization script
python scripts/camera_b_viz_direct.py
