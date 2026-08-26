#!/bin/bash
#SBATCH --job-name=BOEM_bench   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120GB
#SBATCH --time=02:00:00
#SBATCH --output=/home/b.weinstein/logs/BOEM%j.out
#SBATCH --error=/home/b.weinstein/logs/BOEM%j.err
#SBATCH --partition=hpg-b200
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1

# predict_tile(dataloader_strategy="batch") batches IMAGES, not patches: one forward
# pass sees batch_size * 35 patches for this camera. Each case runs in a fresh process
# so one case's fragmentation is not charged to the next as a false OOM.
cd /blue/ewhite/b.weinstein/src/BOEM
IMAGE_DIR=/blue/ewhite/b.weinstein/BOEM/imagery/JPG_20260712_100400

for BATCH in 1 2 4 8 16 24 32 48 64; do
    echo "########## batch_size=${BATCH} ##########"
    uv run python scripts/benchmark_inference.py \
        --image-dir "$IMAGE_DIR" \
        --n-images 60 \
        --batch-sizes "$BATCH" \
        --workers 5 \
        --out "/tmp/bench_b${BATCH}.json"
done
