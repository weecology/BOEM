#!/bin/bash
#SBATCH --job-name=BOEM_wsweep   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120GB
#SBATCH --time=03:00:00
#SBATCH --output=/home/b.weinstein/logs/BOEM%j.out
#SBATCH --error=/home/b.weinstein/logs/BOEM%j.err
#SBATCH --partition=hpg-turin
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1

# Is the DataLoader the bottleneck? Job 39225777 showed throughput on a B200 is inversely
# related to batch size and GPU utilization sits in the single digits, which points at
# JPEG decode + Lustre read rather than the forward pass. This sweeps predict.workers at
# batch_size=1 to test that directly: if the workload is data-bound, workers should move
# the number and the GPU should barely matter.
#
# Submit to both partitions to compare a 24 GB L4 against a 179 GB B200:
#   sbatch submit_worker_sweep.sh                                    # L4  (hpg-turin)
#   sbatch --partition=hpg-b200 submit_worker_sweep.sh               # B200
#
# batch_size stays at 1: batch 4 peaked at 25.5 GB, which does not fit an L4's 24 GB.
# Each case runs in a fresh process so no case inherits another's allocator state.
cd /blue/ewhite/b.weinstein/src/BOEM
IMAGE_DIR=/blue/ewhite/b.weinstein/BOEM/imagery/JPG_20260712_100400

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "cpus available: $(nproc)"

for WORKERS in 0 2 5 10; do
    echo "########## workers=${WORKERS} batch=1 ##########"
    uv run python scripts/benchmark_inference.py \
        --image-dir "$IMAGE_DIR" \
        --n-images 60 \
        --batch-sizes 1 \
        --workers "$WORKERS" \
        --out "/tmp/wsweep_w${WORKERS}.json"
done

# One larger batch for memory headroom reference (expected to OOM on the L4).
echo "########## workers=5 batch=2 ##########"
uv run python scripts/benchmark_inference.py \
    --image-dir "$IMAGE_DIR" \
    --n-images 60 \
    --batch-sizes 2 \
    --workers 5 \
    --out "/tmp/wsweep_b2.json"
