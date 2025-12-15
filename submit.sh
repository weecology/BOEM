#!/bin/bash
#SBATCH --job-name=BOEM   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI ran
#SBATCH --cpus-per-task=5
#SBATCH --mem=200GB
#SBATCH --time=24:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/BOEM%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/BOEM%j.err
#SBATCH --partition=hpg-b200
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1


uv run python main.py image_dir=/blue/ewhite/b.weinstein/BOEM/GulfMexico/JPG_20241220_104800 check_annotations=True active_learning.pool_limit=1000 active_testing.n_images=30 active_learning.n_images=1 debug=False pipeline.gpus=1
