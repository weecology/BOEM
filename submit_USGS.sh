#!/bin/bash
#SBATCH --job-name=BOEM_USGS   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI ran
#SBATCH --cpus-per-task=16
#SBATCH --mem=150GB
#SBATCH --time=48:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/BOEM%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/BOEM%j.err
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4

source activate BOEM

cd ~/BOEM/
python prepare_USGS.py
srun python USGS_backbone.py --batch_size 12 --workers 16

