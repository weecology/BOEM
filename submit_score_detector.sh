#!/bin/bash
#SBATCH --job-name=BOEM_score_det
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=05:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --output=/home/b.weinstein/logs/score_det_%j.out
#SBATCH --error=/home/b.weinstein/logs/score_det_%j.err
# Score one detection checkpoint (any architecture; its recorded input_size is re-applied by detection.load):
#   sbatch submit_score_detector.sh <checkpoint.ckpt> <label>
# 1 VALIDATION   eval_detection_checkpoint.py on test_max_empty_0_25.csv (pooled TP/GT, IoU 0.4, raw preds >= 0.3)
# 2 density      density_strata.py on those predictions, equal precision 0.97 interpolated
# 3 train fit    same two steps on dense_flock/phase0/0d_train_sample.csv (TRAINING tiles: fit, not validation)
# 4 ZERO-SHOT    threshold_sweep.py on the pinned holdout + 2000-frame unscreened queue sample
set -euo pipefail
CKPT=$1; LABEL=$2
cd /blue/ewhite/b.weinstein/src/BOEM/.claude/worktrees/detector-architecture-comparison
export HF_HOME=/blue/ewhite/b.weinstein/.cache/huggingface
PY=.venv/bin/python
OUT=/blue/ewhite/b.weinstein/BOEM/architecture_comparison/scores/$LABEL
mkdir -p $OUT
TRAIN0D=/blue/ewhite/b.weinstein/BOEM/dense_flock/phase0/0d_train_sample.csv
STEM=${CKPT%.ckpt}

# SKIP_VAL=1 reuses step 1's predictions CSV (e.g. after a step-2 crash) instead of re-predicting.
[ -z "${SKIP_VAL:-}" ] && { echo "=== [1] VALIDATION $LABEL ==="; $PY -u scripts/eval_detection_checkpoint.py $CKPT | tee $OUT/validation.txt; }
echo "=== [2] VALIDATION density strata ==="
$PY scripts/density_strata.py $STEM.test_max_empty_0_25_predictions.csv --target-precision 0.97 | tee $OUT/validation_strata.txt
echo "=== [3] TRAINING-tile fit (0d sample) ==="; $PY -u scripts/eval_detection_checkpoint.py $CKPT $TRAIN0D | tee $OUT/train0d.txt
$PY scripts/density_strata.py $STEM.0d_train_sample_predictions.csv $TRAIN0D --target-precision 0.97 | tee $OUT/train0d_strata.txt
echo "=== [4] ZERO-SHOT ==="
$PY -u scripts/threshold_sweep.py --checkpoint $CKPT --label $LABEL --out $OUT/threshold_sweep_$LABEL.json \
    --queue-sample 2000 --batch-size 1 --workers 5
$PY scripts/threshold_sweep.py --report /blue/ewhite/b.weinstein/BOEM/yolo26_comparison/threshold_sweep_c6ecec12_e27.json \
    $OUT/threshold_sweep_$LABEL.json | tee $OUT/zero_shot.txt
cp $STEM.*_predictions.csv $STEM.*_predictions.density_strata.csv $OUT/
