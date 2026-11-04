#!/bin/bash
#SBATCH --job-name=new_camera
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --output=/home/b.weinstein/logs/new_camera%j.out
#SBATCH --error=/home/b.weinstein/logs/new_camera%j.err
#SBATCH --partition=hpg-b200
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1

cd /blue/ewhite/b.weinstein/src/BOEM
export PYTHONPATH=/blue/ewhite/b.weinstein/src/BOEM
uv run python scripts/predict_new_camera.py \
    /blue/ewhite/b.weinstein/BOEM/imagery/new_camera_gull_parent \
    /blue/ewhite/b.weinstein/BOEM/new_camera_test
