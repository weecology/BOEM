#!/bin/bash
#SBATCH --job-name=ML_Workflow_Manager   # Job name
#SBATCH --mail-type=END,FAIL             # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=your.email@example.com  # Where to send mail
#SBATCH --account=your_account_name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --cpus-per-task=4                # Number of CPU cores per task
#SBATCH --mem=16GB                       # Job memory request
#SBATCH --time=24:00:00                  # Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logsML_Workflow_%j.out   # Standard output log
#SBATCH --error=/home/b.weinstein/logs/ML_Workflow_%j.err    # Error log
#SBATCH --partition=gpu                  # Partition name
#SBATCH --gpus=1                         # Number of GPUs (if needed)

# Load any necessary modules
module load python/3.9

# Activate your virtual environment if you're using one
source activate BOEM

# Install or update requirements
pip install -r requirements.txt

# Run tests
pytest tests/

# Run the main Python script
python main.py

# Run the Streamlit app
streamlit run streamlit_app.py --server.port $PORT
