#!/bin/bash
#SBATCH --job-name=searchlight_sub${1}     # Job name includes subject ID
#SBATCH --output=logs/searchlight_sub${1}_%j.out
#SBATCH --error=logs/searchlight_sub${1}_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --partition=main
#SBATCH --qos=standard

# Load modules
module purge
module load MATLAB/2021a
module load spm/12

# Move to home directory
cd /home/$USER

# Run MATLAB script and pass subject ID
matlab -nodisplay -nosplash -r "try, run_decoding_searchlight_curta('$1'), catch e, disp(getReport(e)), exit(1), end, exit(0);"
