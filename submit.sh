#!/bin/bash
#SBATCH --job-name=BOEM   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI ran
#SBATCH --cpus-per-task=1
#SBATCH --mem=150GB
#SBATCH --time=48:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/BOEM%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/BOEM%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=1

source activate BOEM
module load ffmpeg

cd ~/BOEM/
python main.py check_annotations=True active_learning.pool_limit=10 active_testing.n_images=1 active_learning.n_images=1 pipeline_evaluation.debug=True force_training=True
