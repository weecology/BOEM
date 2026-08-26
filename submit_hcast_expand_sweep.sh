#!/bin/bash
#SBATCH --job-name=BOEM_hcast_expand
#SBATCH --mail-type=END
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40GB
#SBATCH --time=04:00:00
#SBATCH --output=/home/b.weinstein/logs/hcast_expand_sweep_%j.out
#SBATCH --error=/home/b.weinstein/logs/hcast_expand_sweep_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

ulimit -c 0
cd /blue/ewhite/b.weinstein/src/BOEM

# Same split the checkpoint was trained/validated on (job 39614374).
SPLIT_DIR="${BOEM_HIER_SPLIT_DIR:-/blue/ewhite/b.weinstein/BOEM/training/classification/checkpoints/buffer_30/a3dc30a085f5442393736ecd96b564c5}"
CHECKPOINT="${BOEM_HCAST_CHECKPOINT:-/blue/ewhite/b.weinstein/src/BOEM/output/usgs_hier/best_checkpoint.pth}"
LABEL_CSV="${BOEM_HCAST_LABEL_CSV:-/blue/ewhite/b.weinstein/src/BOEM/output/usgs_hier/species.csv}"
OUT="${BOEM_HCAST_SWEEP_OUT:-/blue/ewhite/b.weinstein/src/BOEM/output/usgs_hier/expand_sweep.csv}"

uv run python scripts/hcast_expand_sweep.py \
  --checkpoint "$CHECKPOINT" \
  --label-csv "$LABEL_CSV" \
  --train-split-csv "$SPLIT_DIR/usgs_train_split.csv" \
  --val-split-csv "$SPLIT_DIR/usgs_val_split.csv" \
  --image-dir /blue/ewhite/b.weinstein/BOEM/training/crops \
  --taxonomy taxonomy.json \
  --batch-size 64 \
  --workers 4 \
  --out "$OUT" \
  ${BOEM_HCAST_EXTRA:-}
