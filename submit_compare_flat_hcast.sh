#!/bin/bash
#SBATCH --job-name=BOEM_cmp_flat_hcast
#SBATCH --mail-type=END
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40GB
#SBATCH --time=01:00:00
#SBATCH --output=/home/b.weinstein/logs/cmp_flat_hcast_BOEM%j.out
#SBATCH --error=/home/b.weinstein/logs/cmp_flat_hcast_BOEM%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

ulimit -c 0
cd /blue/ewhite/b.weinstein/src/BOEM

uv run python -u scripts/compare_flat_vs_hcast.py \
  --out output/usgs_hier/flat_vs_hcast_val.csv
