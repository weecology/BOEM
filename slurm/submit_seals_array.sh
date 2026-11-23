#!/bin/bash
#SBATCH --job-name=BOEM_seals
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=90GB
#SBATCH --time=08:00:00
#SBATCH --output=/home/b.weinstein/logs/seals_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/seals_%A_%a.err
#SBATCH --partition=hpg-b200
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
# One task per flight (32 flights in seals_flights.txt). Cap concurrency so we don't
# hammer the Label Studio API with 32 simultaneous uploads.
#SBATCH --array=0-31%4

ulimit -c 0
module load ffmpeg

MANIFEST=/blue/ewhite/b.weinstein/src/BOEM/seals_flights.txt
IMAGE_DIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")

if [ -z "$IMAGE_DIR" ]; then
    echo "No flight at manifest line $((SLURM_ARRAY_TASK_ID + 1)) of $MANIFEST" >&2
    exit 1
fi
echo "[seals] task $SLURM_ARRAY_TASK_ID -> $IMAGE_DIR"

# Seals: boem_config sets active_learning.strategy=taxonomy + taxonomy_aliases=[Phocidae],
# which expands to the leaf labels the classifier knows (Halichoerus grypus, Phoca vitulina).
#
# Env: .venv-classification holds the claude/friendly-beaver DeepForest (PR #1334, metadata
# embeddings) that the new classification checkpoint requires. --no-sync so it is used as-is.
# NOTE: classification_model.use_metadata=False is SAFE on the d8995ca8 classifier (verified
# job 37325655): the architecture comes from the checkpoint hparams, and metadata was never
# actually trained in (a flight-name regex matched 0 rows every training run), so the flag only
# skips the lookup and CropModel zero-fills the metadata half. Metadata-less flights (no
# captures.csv) MUST set use_metadata=False — see submit_seals_missing_metadata.sh.
export UV_PROJECT_ENVIRONMENT=/blue/ewhite/b.weinstein/src/BOEM/.venv-classification

# report.enabled=False: generate_report's ctx.add_basemap() downloads Esri tiles and took
# ~2h38m on the pilot (vs ~4 min for the actual prediction work). The Label Studio upload
# happens before the report, so turning it off costs nothing for the seal search.
# flythrough_video off for the same reason — not needed to surface seal candidates.
# predict.min_score=0.5: epoch16 detection scores peak ~0.89 even on obvious animals, so the
# default 0.85 discarded ~95% of valid detections (diagnostic job 37219244). At 0.5, ~66/93
# known seal images retain a detection. Human review in Label Studio filters false positives.
srun uv run --no-sync python main.py \
    image_dir="$IMAGE_DIR" \
    check_annotations=True \
    debug=False \
    predict.min_score=0.5 \
    report.enabled=False \
    flythrough_video.enabled=False \
    active_learning.pool_limit=100000 \
    active_learning.n_images=20 \
    active_testing.n_images=1
