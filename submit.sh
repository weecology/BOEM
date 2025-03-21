#!/bin/bash
#SBATCH --job-name=BOEM   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI ran
#SBATCH --cpus-per-task=10
#SBATCH --mem=200GB
#SBATCH --time=48:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/BOEM%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/BOEM%j.err
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1

source activate BOEM

cd ~/BOEM/

export GDAL_ERROR_ON_LIBJPEG_WARNING=FALSE
export PYTHONPATH=/home/b.weinstein/BOEM:$PYTHONPATH
srun python main.py check_annotations=True active_learning.pool_limit=100 active_testing.n_images=1 active_learning.n_images=1 debug=False pipeline.gpus=1
