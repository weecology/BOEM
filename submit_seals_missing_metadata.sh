#!/bin/bash
#SBATCH --job-name=BOEM_seals_nomd
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180GB
#SBATCH --time=24:00:00
#SBATCH --output=/home/b.weinstein/logs/seals_nomd_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/seals_nomd_%A_%a.err
#SBATCH --partition=hpg-b200
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --array=0-20%4

ulimit -c 0
module load ffmpeg

MANIFEST=/blue/ewhite/b.weinstein/BOEM/detection_diag/missing_metadata_flights.txt
IMAGE_DIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")

if [ -z "$IMAGE_DIR" ]; then
    echo "No flight at manifest line $((SLURM_ARRAY_TASK_ID + 1)) of $MANIFEST" >&2
    exit 1
fi
echo "[seals_nomd] task $SLURM_ARRAY_TASK_ID -> $IMAGE_DIR"

# Metadata-less flights (no captures.csv, or captures with no matching basenames). The seals
# run hard-raised on these. classification_model.use_metadata=False is SAFE on d8995ca8
# (verified job 37325655): architecture comes from checkpoint hparams and metadata was never
# actually trained in, so the flag only skips the lookup (CropModel zero-fills). It also routes
# to the faster batched detection path (metadata_lookup=None), which pairs with batch_size=64.
export UV_PROJECT_ENVIRONMENT=/blue/ewhite/b.weinstein/src/BOEM/.venv-classification

srun uv run --no-sync python main.py \
    image_dir="$IMAGE_DIR" \
    check_annotations=True \
    debug=False \
    classification_model.use_metadata=False \
    predict.min_score=0.5 \
    predict.batch_size=16 \
    predict.workers=4 \
    report.enabled=False \
    flythrough_video.enabled=False \
    active_learning.pool_limit=100000 \
    active_learning.n_images=20 \
    active_testing.n_images=1
