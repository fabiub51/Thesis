#!/bin/bash
#SBATCH --job-name=glmsingle_localizer
#SBATCH --output=logs/glmsingle_localizer_%A_%a.out
#SBATCH --error=logs/glmsingle_localizer_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --array=1-17   

# Load modules
module purge                                   
module load Python/3.12.3-GCCcore-13.3.0

# Define list of subject IDs
subjects=( "01" "02" "03" "04" "05" "06" "07" "08" "10"
           "11" "12" "13" "14" "15" "17" "18" "19" )

subj=${subjects[$SLURM_ARRAY_TASK_ID-1]}

# Activate your virtual environment
source /home/fabiub99/glmenv/bin/activate

# Go to the correct directory
cd /home/$USER/            

# Run your script
python glmsingle_localizer_curta.py $subj