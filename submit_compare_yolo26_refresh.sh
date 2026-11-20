#!/bin/bash
#SBATCH --job-name=BOEM_yolo26_refresh
#SBATCH --mail-type=END
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120GB
#SBATCH --time=04:00:00
#SBATCH --output=/home/b.weinstein/logs/BOEM%j.out
#SBATCH --error=/home/b.weinstein/logs/BOEM%j.err
#SBATCH --partition=hpg-b200
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1

# Redo the YOLO26s comparison (51996709 / 51998927) with a newly trained detector.
# Submit with: sbatch --dependency=afterok:<det_job> --export=ALL,DET_JOB=<det_job> submit_compare_yolo26_refresh.sh
# YOLO26s and 1968e962 JSONs are reused as-is: same pinned holdout, same GT, same recipe.
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/BOEM
OUT=/blue/ewhite/b.weinstein/BOEM/yolo26_comparison
CKPTS=/blue/ewhite/b.weinstein/BOEM/training/checkpoints

# The detection run's checkpoint dir is named by Comet id and not logged, so find the
# one dir whose epoch checkpoints were all written after DET_JOB started.
START=$(sacct -j "${DET_JOB}" -X -n -o Start | head -1 | xargs)
mapfile -t DIRS < <(find "${CKPTS}" -mindepth 2 -maxdepth 2 -name 'epoch*.ckpt' -newermt "${START}" -printf '%h\n' | sort -u)
[ "${#DIRS[@]}" -eq 1 ] || { echo "Expected 1 checkpoint dir newer than ${START}, found ${#DIRS[@]}: ${DIRS[*]}"; exit 1; }
# Lowest val_classification (a loss) -- same rule that picked production 1968e962/epoch29.
CKPT=$(ls "${DIRS[0]}"/epoch*-val_cls*.ckpt | awk -F'val_cls' '{v=$2; sub(/\.ckpt$/,"",v); print v, $0}' | sort -g | head -1 | cut -d' ' -f2-)
LABEL="$(basename "${DIRS[0]}" | cut -c1-8)_$(basename "${CKPT}" | cut -d- -f1 | sed 's/epoch/e/')"
echo "Detection job ${DET_JOB}: sweeping ${CKPT} as ${LABEL}"

uv run python scripts/threshold_sweep.py \
    --checkpoint "${CKPT}" \
    --label "${LABEL}" --out "${OUT}/threshold_sweep_${LABEL}.json" \
    --queue-sample 2000 --batch-size 1 --workers 5

echo; echo "############################################################"; echo
uv run python scripts/threshold_sweep.py --report \
    "${OUT}/threshold_sweep_yolo26s.json" \
    "${OUT}/threshold_sweep_1968e962_e29.json" \
    "${OUT}/threshold_sweep_${LABEL}.json"
