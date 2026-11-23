#!/bin/bash
#SBATCH --job-name=BOEM_zs_size
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --output=/home/b.weinstein/logs/zs_input_size_%j.out
#SBATCH --error=/home/b.weinstein/logs/zs_input_size_%j.err
# ZERO-SHOT (pinned holdout) sweep of production c6ecec12/epoch27 at input 800 (reproduction of
# 52121265), 1000 (native) and 1200 (upsampled): recall/precision on the labelled frames and
# boxes per unscreened frame (the false-alarm side), same recipe as submit_compare_yolo26_refresh.sh.
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/BOEM/.claude/worktrees/detector-architecture-comparison
PY=/blue/ewhite/b.weinstein/src/BOEM/.venv/bin/python
CKPT=/blue/ewhite/b.weinstein/BOEM/training/checkpoints/c6ecec12e078416c9fe645c89bc418cf/epoch27-val_cls0.0867.ckpt
OUT=/blue/ewhite/b.weinstein/BOEM/zero_shot_input_size
mkdir -p $OUT
for S in 800 1000 1200; do
  $PY -u scripts/threshold_sweep.py --checkpoint $CKPT --input-size $S \
      --label c6ecec12_e27_in$S --out $OUT/threshold_sweep_c6ecec12_e27_in$S.json \
      --queue-sample 2000 --batch-size 1 --workers 5
done
$PY scripts/threshold_sweep.py --report /blue/ewhite/b.weinstein/BOEM/yolo26_comparison/threshold_sweep_c6ecec12_e27.json \
    $OUT/threshold_sweep_c6ecec12_e27_in{800,1000,1200}.json
