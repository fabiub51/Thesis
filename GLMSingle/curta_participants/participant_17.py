# Imports 
import sys
sys.path.append('/home/fabiub99/GLMsingle/python')
sys.path.append('/home/fabiub99/fracridge/fracridge')
import numpy as np
import re
import nibabel
import scipy
import scipy.io as sio
from scipy.ndimage import gaussian_filter
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from os.path import join, exists, split
import time
import urllib.request
import warnings
from tqdm import tqdm
from pprint import pprint
warnings.filterwarnings('ignore')
from glmsingle.glmsingle import GLM_single
from scipy.interpolate import PchipInterpolator

# Interpolation Function 
def fmri_interpolate(input_mat, tr_orig=2.0, tr_new=0.5):
    """
    Interpolates 4D fMRI data (X, Y, Z, T) to double the temporal resolution.
    Assumes slice time correction was done to the middle of the TR.

    Parameters:
    - input_mat: 4D numpy array (X, Y, Z, T)
    - tr_orig: Original TR (default 1.0s)
    - tr_new: New TR (default 0.5s)

    Returns:
    - output_mat: 4D numpy array (X, Y, Z, new T)
    """
    assert input_mat.ndim == 4, "Input must be a 4D array."

    num_x, num_y, num_z, num_t = input_mat.shape

    # Create original and new time axes
    original_time = np.arange(0.5, num_t, 1.0) * tr_orig
    new_time = np.arange(0.5, original_time[-1] + tr_new, tr_new)

    # Preallocate the output matrix
    output_mat = np.zeros((num_x, num_y, num_z, len(new_time)), dtype=input_mat.dtype)

    # Interpolate for each voxel
    for x in range(num_x):
        for y in range(num_y):
            for z in range(num_z):
                time_series = input_mat[x, y, z, :]
                if np.any(time_series):  # Only interpolate if there is non-zero data
                    interpolator = PchipInterpolator(original_time, time_series)
                    output_mat[x, y, z, :] = interpolator(new_time)

    return output_mat

# Set up directories 
homedir = '/home/fabiub99'  # fest definieren
datadir = join(homedir,'miniblock','derivatives')
presdir = join(homedir, 'designmats')

# Set up variables
runtypes = ['sus', 'er', 'miniblock']
sm_fwhm = 2 # voxels 
tr_old = 2 # before resampling
tr_new = 0.5 # after resampling
subjects = ["17"] # subjects 9 and 16 excluded due to motion

for sub in subjects: 
    partString = f'P0{sub}'
    subString = f'sub-{sub}'

    for runTypeIdx in runtypes:
        for smoothing in range(2):

            # Specify stimulus duration 
            if runTypeIdx == 'er':
                stimdur = 1
            else: 
                stimdur = 4

            # get files for design matrices and fmri data
            pattern = presdir + f'/P0{sub}_ConditionRich_Run*_{runTypeIdx}.csv'
            matches = glob.glob(pattern)
            matches.sort()
            
            # store data and designs
            data = []
            design = []
            for i in range(len(matches)):
                designMat = pd.read_csv(matches[i], header=None)
                num = re.search(r'Run_(\d+)', matches[i])
                # reminder: the runNum of the designmats is not the same as for the functional data as localizer runs were interspersed 
                runNum = int(num.group(1))
                if (runNum >3) & (runNum < 7) & (sub != '01'): 
                    runNum += 1 # localizer run after 3rd functional run
                elif (runNum >= 7):
                    runNum += 2 # localizer run after 6th functional run (7th overall)
                elif (sub == '01') & (runNum >4) & (runNum < 7):
                    runNum +=1
                
                # get the nii.gz file 
                if runNum < 10: 
                    file_path = join(datadir, subString, 'func', subString + f'_task-func_run-0{runNum}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
                else: 
                    file_path = join(datadir, subString, 'func', subString + f'_task-func_run-{runNum}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
                
                # Load the file 
                origData = nibabel.load(file_path)
                # Interpolation 
                interpData = fmri_interpolate(origData.get_fdata(), tr_old, tr_new)

                # Resampling the Design Matrix 
                upsample_factor = 2
                time_points, conditions = designMat.shape

                upsampled_matrix = np.zeros((time_points * upsample_factor, conditions))

                # Fill design matrix with values and remove the first row 
                upsampled_matrix[::upsample_factor, :] = designMat
                design.append(upsampled_matrix[1:-1, :])
                

                if smoothing == 1: 
                    
                    sigma = sm_fwhm/np.sqrt(8 * np.log(2))
                    numX,numY,numz,numT = interpData.shape
                    smoothedData = np.full(interpData.shape, np.nan)
                    for tIdx in range(numT):
                        cData = interpData[:,:,:,tIdx]
                        smoothedData[:,:,:,tIdx] = gaussian_filter(cData, sigma)
                    data.append(smoothedData)

                    outputdir = join(homedir, 'miniblock', 'Outputs', subString ,f'sm_{sm_fwhm}_vox_{subString}_{runTypeIdx}')
                
                else: 
                    data.append(interpData)
                    outputdir = join(homedir, 'miniblock', 'Outputs', subString , f'unsmoothed_{subString}_{runTypeIdx}')
                    
            opt = dict()
            opt['wantlibrary'] = 1
            opt['wantglmdenoise'] = 1
            opt['wantfracridge'] = 1
            opt['wantfileoutputs'] = [1, 1, 1, 1]
            opt['wantmemoryoutputs'] = [0, 0, 0, 0]


            glmsingle_obj = GLM_single(opt)

            pprint(glmsingle_obj.params)

            start_time = time.time()

            os.makedirs(outputdir, exist_ok=True)
            results_glmsingle = glmsingle_obj.fit(
                design, 
                data, 
                stimdur, 
                tr_new, 
                outputdir=outputdir
            )

            elapsed_time = time.time() - start_time
            print('\telapsed_time: '
            f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
            )               