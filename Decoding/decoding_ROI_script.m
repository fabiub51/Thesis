accuracies = [];

% Cell arrays of character vectors ({} not [])
designs = {'miniblock', 'sus', 'er'};
subjects = {'01', '02', '03', '04', '05', '06', '07', '08', '10', '11', '12', '13', '14', '15', '17', '18', '19'};
mask_dir = '/Users/danieljanini/Documents/Thesis/miniblock/derivatives/';

basedir = '/Users/danieljanini/Documents/Thesis/miniblock/Outputs';  

output_dir = 'Occipital';

for d = 1:length(designs)
    design = designs{d};  % get string from cell array
    for s = 1:length(subjects)
        subject = strcat('sub-', subjects{s});

        mask = fullfile(mask_dir, subject,'anat','occipital_mask_sm_2_vox.nii');
        
        % Run your decoding function
        run_decoding(subject, design, 'sm_2_vox', mask, output_dir);
        
        % Load the confusion matrix
        directory = fullfile(basedir, 'decoding','ROI', output_dir, design, subject);
        conf_matrix = load(fullfile(directory, 'res_confusion_matrix.mat'));
        
        % Calculate accuracy
        M = conf_matrix.results.confusion_matrix.output{1,1};

        % Save the matrix as a 2D array in a .mat file
        save(fullfile(directory,'confusion_matrix.mat'), 'M');

        accuracy = trace(M) / sum(M(:));
        fprintf('Overall accuracy for %s (%s): %.2f%%\n', subject, design, accuracy * 100);
        
        % Store accuracy
        accuracies(end+1) = accuracy;
    end
end
