#!/bin/bash
#SBATCH --job-name=15_permutation            # Descriptive job name
#SBATCH --output=logs/permutation_15_%j.out         # Store logs in a folder, include Job ID
#SBATCH --error=logs/permutation_15_%j.err          # Capture stderr in a separate file
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                      # Use multiple cores if your script can leverage it
#SBATCH --time=02:00:00                        # More realistic time for GLM fitting
#SBATCH --mem=16G                              # Increase memory for fMRI data, safer for large jobs
#SBATCH --partition=main
#SBATCH --qos=standard

# Load modules
module purge
module load Python/3.12.3-GCCcore-13.3.0
source ~/my_env_py312/bin/activate

# Go to the correct directory
cd /home/$USER/          

# Run your script
python reliability_permutations_15.py