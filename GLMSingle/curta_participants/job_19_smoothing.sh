#!/bin/bash
#SBATCH --job-name=glmsingle_beta              # Descriptive job name
#SBATCH --output=logs/glmsingle_%j.out         # Store logs in a folder, include Job ID
#SBATCH --error=logs/glmsingle_%j.err          # Capture stderr in a separate file
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                      # Use multiple cores if your script can leverage it
#SBATCH --time=7:00:00                        # More realistic time for GLM fitting
#SBATCH --mem=64G                              # Increase memory for cufMRI data, safer for large jobs
#SBATCH --partition=main
#SBATCH --qos=standard

# Load modules
module purge                                   
module load Python/3.12.3-GCCcore-13.3.0

# Activate your virtual environment
source /home/fabiub99/glmenv/bin/activate

# Go to the correct directory
cd /home/$USER/            

# Run your script
python participant_19_smoothing.py
