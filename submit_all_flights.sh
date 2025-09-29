#!/bin/bash

GULF_DIR="/blue/ewhite/b.weinstein/BOEM/GulfMexico"
ATLANTIC_DIR="/blue/ewhite/b.weinstein/BOEM/Atlantic"

SLEEP_SEC=3

for ROOT in "$ATLANTIC_DIR" "$GULF_DIR"; do
  for folder in "$ROOT"/*/; do
    folder="${folder%/}"
    echo "Submitting job with image_dir: $folder"
    sbatch --export=ALL,IMAGE_DIR="$folder" <<EOF
#!/bin/bash
#SBATCH --job-name=BOEM
#SBATCH --mail-type=END
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=70GB
#SBATCH --time=48:00:00
#SBATCH --output=/home/b.weinstein/logs/BOEM_%j.out
#SBATCH --error=/home/b.weinstein/logs/BOEM_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

uv run python main.py image_dir=\$IMAGE_DIR check_annotations=True active_learning.pool_limit=100000 debug=False pipeline.gpus=1
EOF
    sleep "$SLEEP_SEC"
  done
done
