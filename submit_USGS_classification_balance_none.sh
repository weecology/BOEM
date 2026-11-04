#!/bin/bash
#SBATCH --job-name=BOEM_USGS_bfnone   # Job name
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

# Bracket run: no class-balance downsampling at all (full 176,847 train crops,
# natural class imbalance) vs the 4x baseline from job 40994173 (176,847 -> 59,492,
# micro 0.7651 / macro 0.6530) and the 2x arm in submit_USGS_classification_balance_factor2.sh.
# Full pool is ~3x larger than the 4x-cap run, so expect a longer wall clock per epoch.
# See JOB_LEDGER.md for the full bracket writeup.

# prepare_USGS already run separately
# uv run python -u scripts/prepare_USGS.py

export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO

# Isolated env: classification needs the claude/friendly-beaver DeepForest branch (PR #1334, metadata_csv),
# while detection trains against the shared .venv (tmp/hpc-balanced-empty-frames). Keep them separate so
# this job's uv sync does not mutate the detection job's environment.
export UV_PROJECT_ENVIRONMENT=/blue/ewhite/b.weinstein/src/BOEM/.venv-classification
# use_metadata=False to match the 40994173 baseline exactly (only class_balance_factor differs).
uv run python -u /blue/ewhite/b.weinstein/src/BOEM/scripts/USGS_classification.py \
    classification_model.use_metadata=False \
    classification_model.class_balance_factor=null
