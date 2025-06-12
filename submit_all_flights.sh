#!/bin/bash
# filepath: /home/b.weinstein/BOEM/submit_all_gulfmexico.sh

GULF_DIR="/blue/ewhite/b.weinstein/BOEM/GulfMexico"

for folder in "$GULF_DIR"/*/; do
    folder="${folder%/}"  # Remove trailing slash
    echo "Submitting job with image_dir: $folder"
    sbatch --export=ALL,IMAGE_DIR="$folder" <<EOF
#!/bin/bash
#SBATCH --job-name=BOEM
#SBATCH --mail-type=END
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60GB
#SBATCH --time=24:00:00
#SBATCH --output=/home/b.weinstein/logs/BOEM_%j.out
#SBATCH --error=/home/b.weinstein/logs/BOEM_%j.err
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=a100:1

source activate BOEM

cd ~/BOEM/

echo "Running with image_dir: \$IMAGE_DIR"

export GDAL_ERROR_ON_LIBJPEG_WARNING=FALSE
export PYTHONPATH=/home/b.weinstein/BOEM:\$PYTHONPATH
srun python main.py image_dir=\$IMAGE_DIR check_annotations=True active_learning.pool_limit=100000 active_testing.n_images=5 active_learning.n_images=50 debug=False pipeline.gpus=1
EOF
done
