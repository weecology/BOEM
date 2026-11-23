#!/bin/bash
#SBATCH --job-name=BOEM_neaq
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
# 350GB, not the 90GB the seals array used. neaq frames are 9504x6336 (whale) and
# 8688x5792 (belly) — ~2x the ~6464x4852 UBFAI images that request was sized for — and
# without a metadata_lookup detection.py takes predict_tile's batched path, where
# dataloader_strategy="batch" holds a whole frame per worker. Pilot 37329130 peaked at
# 83.7GB and was OOM-killed at 90GB. b200 nodes have 2TB, so headroom is cheap.
#SBATCH --mem=350GB
#SBATCH --time=12:00:00
#SBATCH --output=/home/b.weinstein/logs/neaq_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/neaq_%A_%a.err
#SBATCH --partition=hpg-b200
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
# 40 leaf dirs (20 belly + 20 whale) in neaq_flights.txt, whale-first so the small/fast
# tasks surface problems before the 70k-image belly cameras commit real GPU time.
# Cap concurrency so we don't hammer the Label Studio API.
#SBATCH --array=0-39%4

ulimit -c 0
module load ffmpeg

MANIFEST=/blue/ewhite/b.weinstein/src/BOEM/neaq_flights.txt
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")

if [ -z "$LINE" ]; then
    echo "No flight at manifest line $((SLURM_ARRAY_TASK_ID + 1)) of $MANIFEST" >&2
    exit 1
fi

# Manifest is TAB separated: <flight_name>\t<image_dir>. neaq paths contain spaces
# ("Belly camera edited"), so split on tab only and quote every expansion.
FLIGHT_NAME=$(printf '%s' "$LINE" | cut -f1)
IMAGE_DIR=$(printf '%s' "$LINE" | cut -f2)

if [ ! -d "$IMAGE_DIR" ]; then
    echo "[neaq] image_dir does not exist: $IMAGE_DIR" >&2
    exit 1
fi
echo "[neaq] task $SLURM_ARRAY_TASK_ID -> $FLIGHT_NAME : $IMAGE_DIR"

export UV_PROJECT_ENVIRONMENT=/blue/ewhite/b.weinstein/src/BOEM/.venv-classification

# flight_name: every neaq date has a leaf dir literally named "Belly camera edited", so the
# image_dir basename is not unique. Without this override all 20 dates share one crop dir,
# checkpoint dir and prediction cache and silently overwrite each other.
#
# use_metadata=False: neaq has no flight metadata CSVs in report.metadata_dir. The classifier
# (d8995ca8, metadata_dim=32) keeps its architecture either way — pipeline.py loads it via
# load_from_checkpoint, which restores use_metadata from the ckpt hparams, not from this flag.
# The flag only skips the lookup, so CropModel.forward zero-fills the metadata half of the
# classifier input (deepforest model.py:522-526). Verified by scripts/verify_metadata_fallback.py.
#
# predict.batch_size=4. batch_size counts *images*, not patches: detection.py uses
# dataloader_strategy="batch" -> the MultiImage dataset, whose __len__ is the number of paths
# and whose collate_fn flattens every crop, so one forward pass sees batch_size * patches-per-
# image. These frames are 9504x6336 -> ~70 patches, so 4 images ~= 280 patches ~= 11 GB, while
# the old default of 64 meant 4,480 patches: pilot 37330544 died asking cuDNN for 170.90 GiB
# on an 178 GiB B200. boem_config's default is now 1 (benchmarked fastest and leanest, job
# 39225777); 4 is kept here only because it is the value this flight's runs were done at.
# (With a metadata_lookup the strategy is still "batch" — detection.py just passes one path at
# a time, so the DataLoader yields a single image and batch_size has no effect on that path.)
#
# report/flythrough off: not needed to get predictions into Label Studio, and generate_report
# was the dominant cost per task.
srun uv run --no-sync python main.py \
    image_dir="$IMAGE_DIR" \
    flight_name="$FLIGHT_NAME" \
    check_annotations=True \
    debug=False \
    predict.min_score=0.5 \
    predict.batch_size=4 \
    classification_model.use_metadata=False \
    report.enabled=False \
    flythrough_video.enabled=False \
    active_learning.pool_limit=100000 \
    active_learning.n_images=20 \
    active_testing.n_images=1
