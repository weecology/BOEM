#!/bin/bash
#SBATCH --job-name=BOEM_winsweep
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

# Year-2 annual report: measure recall / precision / empty-frame rate at BOTH ends of the
# window (2025-09-16 .. 2026-09-15) on the SAME pinned zero-shot holdout, so the report has a
# like-for-like start-vs-end row. No such measurement exists today: Comet shows only 82 of
# 1,233 in-window experiments log any eval metric, and the earliest logging box_precision /
# box_recall / empty_frame_accuracy is 2026-05-27.
#
# START of window: ee7ba70a, trained 2025-09-18 (Comet val_classification 0.0523), the best
#   of 52 detection checkpoints written 2025-09-16..10-15. NOTE this is a contemporaneous
#   pipeline-trained model, NOT the one boem_config.yaml pointed at on 2025-09-30
#   (f3c687c8824f4b9fafa4bafbbe3fad8b.pl) -- that file no longer exists on /blue.
#   Holdout is clean by construction: both Feb-2026 holdout flights postdate this checkpoint.
#
# END of window: 1968e962/epoch29 (job 40994172, 2026-09-03), best val_cls 0.0899 of 30
#   epochs. Holdout clean by construction: trained after the 2026-08-06 pinning, which
#   explicitly excludes JPG_20260201_134000 + JPG_20260202_141900.
#   This ALSO fills the still-blank "Refined Model" column of the survey metrics doc.
#
# batch_size=1 and workers=5 are the optimum measured by worker sweep 39272218.
# a09c6933/epoch16 is deliberately NOT re-run: threshold_sweep_a09c6933_e16.json (2026-08-13)
# already holds it, and it is the contamination control -- it was trained 2026-07-07, a month
# BEFORE the holdout was pinned, so its split is undocumented and may include these flights.

cd /blue/ewhite/b.weinstein/src/BOEM
CKPT_DIR=/blue/ewhite/b.weinstein/BOEM/training/checkpoints

uv run python scripts/threshold_sweep.py \
    --checkpoint "/blue/ewhite/b.weinstein/BOEM/detection/checkpoints/JPG_20241219_165600/ee7ba70a6bdf47ae99eab2b8f3024bce.ckpt" \
    --label window_start_ee7ba70a \
    --queue-sample 2000 --batch-size 1 --workers 5

echo
echo "############################################################"
echo

uv run python scripts/threshold_sweep.py \
    --checkpoint "${CKPT_DIR}/1968e962113f4c57bc6bf5f07fae1b25/epoch29-val_cls0.0899.ckpt" \
    --label window_end_1968e962_e29 \
    --queue-sample 2000 --batch-size 1 --workers 5
