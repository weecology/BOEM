#!/bin/bash
#SBATCH --job-name=BOEM_det_aug
#SBATCH --mail-type=END
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
# Bumped from 200GB: job 38309080 reported sstat MaxRSS 244GB over the 200GB request and
# survived, and this crop set is ~4.5% larger (608,980 vs 582,531 train rows). Cheap
# insurance against an OOM 12h into a 12.5h run.
#SBATCH --mem=300GB
#SBATCH --time=24:00:00
#SBATCH --output=/home/b.weinstein/logs/detection_BOEM_aug_%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/detection_BOEM_aug_%x_%j.err
#SBATCH --partition=hpg-b200
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1

ulimit -c 0

export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO

POSITIVE_BATCH_FRACTION="${POSITIVE_BATCH_FRACTION-0.90}"

EXTRA_ARGS=()
if [[ -n "$POSITIVE_BATCH_FRACTION" && "$POSITIVE_BATCH_FRACTION" != "null" && "$POSITIVE_BATCH_FRACTION" != "none" ]]; then
    EXTRA_ARGS+=(--positive-batch-fraction "$POSITIVE_BATCH_FRACTION")
    FRAC_LABEL="posfrac${POSITIVE_BATCH_FRACTION}"
else
    FRAC_LABEL="random"
fi

EXP_NAME="${EXP_NAME:-detection_${FRAC_LABEL}_${SLURM_JOB_ID}}"
export EXP_NAME

echo "EXP_NAME=$EXP_NAME"
echo "POSITIVE_BATCH_FRACTION=${POSITIVE_BATCH_FRACTION:-<unset, random sampling>}"

srun uv run python scripts/USGS_backbone.py \
    --batch_size 64 \
    --workers 32 \
    --lr 0.001 \
    --epochs 10 \
    "${EXTRA_ARGS[@]}" \
    --exp-name "$EXP_NAME"
