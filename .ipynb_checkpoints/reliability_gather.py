import numpy as np 
import nibabel as nib
from nilearn import plotting,image
import pandas as pd 
import os 
import glob 
from os.path import join, exists, split
from scipy.stats import pearsonr
import scipy.ndimage
from statsmodels.stats.multitest import fdrcorrection
import re

outdir = '/Users/danieljanini/Documents/Thesis/miniblock/Outputs/'
homedir = split(os.getcwd())[0]
datadir = '/Users/danieljanini/Documents/Thesis/miniblock/'
presdir = join(homedir, 'Behavior', 'designmats')
subs = ["01", "02"]
smooths = ["sm_2_vox", "unsmoothed"]
runtypes = ['miniblock', "er", "sus"]

results = []

for sub in subs: 
    for runtype in runtypes:
        for smoothing in smooths: 

            results_glmsingle = dict()
            results_glmsingle['typed'] = np.load(join(outdir,f'sub-{sub}',f'{smoothing}_sub-{sub}_{runtype}/TYPED_FITHRF_GLMDENOISE_RR.npy'), allow_pickle=True).item()
            betas = results_glmsingle['typed']['betasmd']

            brain_mask = image.load_img(join(datadir, 'derivatives', f'sub-{sub}', 'anat', 'visual_voxels_mask_sm_2_vox.nii.gz'))
            print("Mask shape:", brain_mask.shape)
            print("Mask affine:", brain_mask.affine)
            mask = brain_mask.get_fdata()
            print("Mask unique values:", np.unique(mask))
            masked_betas = betas[mask.astype(bool)]
            print("Masked beta shape:", masked_betas.shape)  
            unmasked_betas = np.zeros(betas.shape)
            unmasked_betas[mask.astype(bool)] = masked_betas
            print("Masked betas reverted to original size: ", unmasked_betas.shape)

            pattern = presdir + f'/P0{sub}_ConditionRich_Run*_{runtype}.csv'
            matches = glob.glob(pattern)
            matches.sort()
            design = []
            for i in range(len(matches)):
                designMat = pd.read_csv(matches[i], header=None)
                print(f"Size of Design Matrix before upsampling: {designMat.shape}")
                num = re.search(r'Run_(\d+)', matches[i])
                # reminder: the runNum of the designmats is not the same as for the functional data as localizer runs were interspersed 
                runNum = int(num.group(1))
                if (runNum >3) & (runNum < 7) & (sub != '01'): 
                    runNum += 1 # localizer run after 3rd functional run
                elif (runNum >= 7):
                    runNum += 2 # localizer run after 6th functional run (7th overall)
                elif (sub == '01') & (runNum >4) & (runNum < 7):
                    runNum +=1
                design.append(designMat)

            all_design = np.vstack((design[0], design[1], design[2]))
            condition_mask = all_design.sum(axis=1) > 0
            condition_vector = np.argmax(all_design[condition_mask], axis=1)
            assert(condition_vector.shape == (240,))

            n_conditions = 40
            max_reps = 6

            repindices = np.full((max_reps, n_conditions), np.nan)

            # Fill in the indices
            for p in range(n_conditions):  
                inds = np.where(condition_vector == p)[0]  
                repindices[:len(inds), p] = inds  

            # Initialize output
            X, Y, Z, T = unmasked_betas.shape
            n_reps, n_conds = repindices.shape
            betas_per_condition = np.full((X, Y, Z, n_reps, n_conds), np.nan)

            # Loop through conditions
            for cond in range(n_conds):
                trial_indices = repindices[:, cond]
                
                for rep, trial_idx in enumerate(trial_indices):
                    if not np.isnan(trial_idx):
                        trial_idx = int(trial_idx)  # convert from float to int
                        betas_per_condition[:, :, :, rep, cond] = unmasked_betas[:, :, :, trial_idx]

            reliability_map = np.full((X, Y, Z), np.nan)
            for x in range(X):
                for y in range(Y):
                    for z in range(Z):
                        # Extract betas for this voxel across repetitions (shape: (n_reps, n_conds))
                        voxel_betas = betas_per_condition[x, y, z, :, :]
                        
                        # Split into even and odd repetitions
                        even_betas_mean = np.mean(voxel_betas[::2, :], axis=0) 
                        odd_betas_mean = np.mean(voxel_betas[1::2, :], axis=0)

                        
                        corr, _ = pearsonr(even_betas_mean, odd_betas_mean)
                        
                        # Take the mean correlation across conditions (or use another aggregation method)
                        reliability_map[x, y, z] = corr
            reliability = np.nanmedian(reliability_map[:,:44,:])
            print(reliability)

            results.append(
                {
                "subject": sub,
                "runtype": runtype,
                "smoothing": smoothing,
                "reliability": reliability
            }
            )
