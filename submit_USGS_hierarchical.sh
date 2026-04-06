#!/bin/bash
#SBATCH --job-name=BOEM_USGS_hier
#SBATCH --mail-type=END
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60GB
#SBATCH --time=48:00:00
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
ANNOTATIONS_DIR="/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/crops"
IMAGE_DIR="/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/crops"
CHECKPOINT_DIR="/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/classification/checkpoints"
OUTPUT_DIR="${BOEM_OUTPUT_DIR:-output/usgs_hier}"
TAXONOMY="taxonomy.json"
# Required: set to e.g. "$CHECKPOINT_DIR/buffer_30/<comet_id>" from USGS_classification run
SPLIT_DIR="${BOEM_HIER_SPLIT_DIR}"

cd /blue/ewhite/b.weinstein/src/BOEM
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
  --expand-pixels 30 \
  --num-workers 4
