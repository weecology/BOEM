#!/bin/bash
#SBATCH --job-name=BOEM_dense_p0
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --time=06:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --output=/home/b.weinstein/logs/dense_flock_p0_%j.out
#SBATCH --error=/home/b.weinstein/logs/dense_flock_p0_%j.err
# Dense-flock Phase 0 (inference only, VALIDATION tiles): paste test, label audit,
# resolution/NMS re-predictions. Outputs: /blue/ewhite/b.weinstein/BOEM/dense_flock/phase0/
#   sbatch submit_dense_flock_phase0.sh [checkpoint.ckpt]
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/BOEM
uv run python -u scripts/dense_flock_phase0.py "$@"
