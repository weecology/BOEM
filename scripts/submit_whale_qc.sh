#!/bin/bash
#SBATCH --job-name=BOEM_whale_qc
#SBATCH --mail-type=END
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40GB
#SBATCH --time=4:00:00
#SBATCH --output=/home/b.weinstein/logs/whale_qc%j.out
#SBATCH --error=/home/b.weinstein/logs/whale_qc%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hpg-default

cd /blue/ewhite/b.weinstein/src/BOEM

uv run python scripts/upload_class_to_review.py "Stenella frontalis" --min-test-images 1
uv run python scripts/upload_class_to_review.py "Delphinus delphis" --min-test-images 1
uv run python scripts/upload_class_to_review.py "Balaenoptera acutorostrata" --min-test-images 1
uv run python scripts/upload_class_to_review.py "Balaenoptera physalus" --min-class-count 10 --min-test-images 1
uv run python scripts/upload_class_to_review.py "Megaptera novaeangliae" --min-class-count 10 --min-test-images 1
