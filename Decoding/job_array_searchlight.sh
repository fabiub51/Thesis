#!/bin/bash
#SBATCH --job-name=searchlight_array
#SBATCH --output=logs/searchlight_%A_%a.out
#SBATCH --error=logs/searchlight_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=33:00:00
#SBATCH --mem=8G
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --array=1-17   

# Load modules
module purge
module load MATLAB/2021a
module load spm/12

# Define list of subject IDs
subjects=( "01" "02" "03" "04" "05" "06" "07" "08" "10"
           "11" "12" "13" "14" "15" "17" "18" "19" )

# Pick subject based on SLURM_ARRAY_TASK_ID
subj=${subjects[$SLURM_ARRAY_TASK_ID-1]}

# Go to home directory
cd /home/$USER

# Run MATLAB script and pass subject
matlab -nodisplay -nosplash -r "try, run_decoding_searchlight_curta('$subj'), catch e, disp(getReport(e)), exit(1), end, exit(0);"
