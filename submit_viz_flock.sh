#!/bin/bash
#SBATCH --job-name=BOEM_viz_flock
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=00:40:00
#SBATCH --output=/home/b.weinstein/logs/viz_flock_%j.out
#SBATCH --error=/home/b.weinstein/logs/viz_flock_%j.err
#SBATCH --partition=hpg-b200
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1

ulimit -c 0

srun uv run python scripts/viz_flock_boxes.py \
  --checkpoint /blue/ewhite/b.weinstein/BOEM/training/checkpoints/a09c69331af8496380cbf99e3859d656/epoch16-val_cls0.0163.ckpt \
  --gt-csv /blue/ewhite/b.weinstein/BOEM/detection_cap_probe/feb2022/train_annotations.csv \
  --image-dir "/blue/ewhite/b.weinstein/BOEM/UBFAI Data Collection/imagery" \
  --images C7_L7_F2981_T20220210_155152_128.JPG C1_L28_F7747_T20220211_121840_687.JPG \
  --out-dir /blue/ewhite/b.weinstein/BOEM/detection_cap_probe/viz \
  --patch 1000 --predict-score 0.05 --production-score 0.70
