#!/bin/bash
#SBATCH --job-name=BOEM_score_1000
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --time=10:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --output=/home/b.weinstein/logs/score_input1000_%j.out
#SBATCH --error=/home/b.weinstein/logs/score_input1000_%j.err
# Score native-1000 detector e0cb9f95/epoch29 (52502097) at 1000 on val strata, 0d train fit, zero-shot, July load;
# plus c6ecec12 at 1000 (min=max) on the train fit and July load. Outputs: /blue/ewhite/b.weinstein/BOEM/dense_flock/score_at_size/
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/BOEM/.claude/worktrees/dense-flock
CK=/blue/ewhite/b.weinstein/BOEM/training/checkpoints
uv run python -u scripts/score_detector_at_size.py \
    e0cb9f95_e29_in1000 $CK/e0cb9f957c404690becdd370ab6938fb/epoch29-val_cls0.0799.ckpt 1000 \
    c6ecec12_e27_in1000:train,july $CK/c6ecec12e078416c9fe645c89bc418cf/epoch27-val_cls0.0867.ckpt 1000
