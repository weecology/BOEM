#!/bin/bash
#SBATCH --job-name=BOEM_USGS_hier
#SBATCH --mail-type=END
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60GB
# 100 epochs at ~28 min/epoch is ~47h, which 48h did not safely cover: 39614374 needed
# 41h42m and the current train set is larger (179,569 rows vs 176,338). The partition
# allows 14 days and there is no --resume, so a wall-clock kill costs the whole run.
#SBATCH --time=96:00:00
#SBATCH --output=/home/b.weinstein/logs/classification_hier_BOEM%j.out
#SBATCH --error=/home/b.weinstein/logs/classification_hier_BOEM%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

ulimit -c 0

# Same data source as USGS_classification. For direct comparability (same test set
# as CropModel), FIRST run USGS_classification.py and set SPLIT_DIR to that run's
# split dir (checkpoint_dir/buffer_30/<comet_id>) containing usgs_train_split.csv
# and usgs_val_split.csv.
ANNOTATIONS_DIR="/blue/ewhite/b.weinstein/BOEM/training/crops"
IMAGE_DIR="/blue/ewhite/b.weinstein/BOEM/training/crops"
CHECKPOINT_DIR="/blue/ewhite/b.weinstein/BOEM/training/classification/checkpoints"
OUTPUT_DIR="${BOEM_OUTPUT_DIR:-output/usgs_hier}"
TAXONOMY="taxonomy.json"
EXPAND_PIXELS="${BOEM_EXPAND_PIXELS:-30}"

cd /blue/ewhite/b.weinstein/src/BOEM

# Auto-discover the most recent split from USGS_classification for the matching buffer size.
# Override by setting BOEM_HIER_SPLIT_DIR explicitly.
if [ -n "${BOEM_HIER_SPLIT_DIR}" ]; then
    SPLIT_DIR="${BOEM_HIER_SPLIT_DIR}"
else
    SPLIT_DIR=$(find "${CHECKPOINT_DIR}/buffer_${EXPAND_PIXELS}" -name "usgs_train_split.csv" -printf "%T@ %h\n" 2>/dev/null | sort -n | tail -1 | awk '{print $2}')
fi

if [ -z "${SPLIT_DIR}" ] || [ ! -f "${SPLIT_DIR}/usgs_train_split.csv" ]; then
    echo "ERROR: No usgs_train_split.csv found under ${CHECKPOINT_DIR}/buffer_${EXPAND_PIXELS}."
    echo "Run USGS_classification.py first, or set BOEM_HIER_SPLIT_DIR to the split directory."
    exit 1
fi

echo "Using CropModel split from $SPLIT_DIR (same test data) and extending train with higher-taxon labels"
uv run python scripts/USGS_hierarchical.py \
  --train-split-csv "$SPLIT_DIR/usgs_train_split.csv" \
  --val-split-csv "$SPLIT_DIR/usgs_val_split.csv" \
  --annotations-dir "$ANNOTATIONS_DIR" \
  --image-dir "$IMAGE_DIR" \
  --taxonomy "$TAXONOMY" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size 32 \
  --epochs 100 \
  --expand-pixels "${EXPAND_PIXELS}" \
  --num-workers 4
