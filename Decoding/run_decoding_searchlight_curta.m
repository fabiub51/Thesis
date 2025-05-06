function run_decoding_searchlight_curta(subj)
% run_decoding_searchlight_curta(subj)
% Pass subject ID (e.g., '01') as argument

if nargin < 1
    error('You must pass subj as argument.');
end

designs= {'miniblock', 'sus', 'er'};
smoothing = 'sm_2_vox';
basedir = '/home/fabiub99/miniblock/Outputs/';
group_mask = '/home/fabiub99/miniblock/decoding/group_mask.nii';
pres_dir = '/home/fabiub99/designmats';

addpath(genpath('/home/fabiub99/decoding_toolbox'));
spm('defaults', 'fmri');
spm_jobman('initcfg');

parpool('local', 4);  % Start parallel pool

for design = designs
    design = char(design);

    cfg = decoding_defaults;
    cfg.analysis = 'searchlight';  
    cfg.decoding.software = 'libsvm';  
    cfg.decoding.method = 'classification';

    beta_folder = fullfile(basedir, strcat(smoothing, '_sub-',subj,'_',design), 'nifti_betas');
    cfg.results.dir = fullfile(basedir, 'decoding', 'searchlight', design, subj);
    if ~exist(cfg.results.dir, 'dir')
        mkdir(cfg.results.dir);  
    end

    n_betas = 240;
    beta_filenames = strcat('beta_', sprintfc('%04d', 1:n_betas), '.nii');
    cfg.files.name = fullfile(beta_folder, beta_filenames);
    cfg.files.name = transpose(cfg.files.name);  

    cfg.files.mask = group_mask;

    cfg.results.output = {'accuracy_minus_chance'};

    sub_index = subj(end-1:end);
    pattern = strcat('^P0',sub_index,'_.*_',design,'\.csv$');  
    files = dir(fullfile(pres_dir, '*.csv'));
    filenames = {files.name};

    matches = ~cellfun('isempty', regexp(filenames, pattern));
    design_files = fullfile(pres_dir, filenames(matches));

    labels = [];
    for i = 1:length(design_files)
        data = readtable(design_files{i});
        [rows, cols] = size(data);
        for r = 1:rows
            for c = 1:cols
                if data{r,c} == 1
                    labels(end+1) = c;  
                end
            end
        end
    end

    cfg.files.label = labels(:);  

    unique_labels = unique(labels);
    chunks = zeros(size(labels));
    for i = 1:length(unique_labels)
        condition_idx = find(labels == unique_labels(i));
        chunks(condition_idx) = 1:length(condition_idx);
    end

    cfg.files.chunk = chunks;
    cfg.searchlight.radius = 3;

    cfg.design.function = 'make_design_cv';
    cfg.results.overwrite = 1;
    cfg.verbose = 0;
    cfg.scale.method = 'min0max1global'; 
    cfg.scale.estimation = 'all';
    cfg.design = make_design_cv(cfg); 

    cfg.design.fig = 0;
    
    % NEW: use decoding toolbox parallelization
    cfg.parallel.nproc = 4;

    decoding(cfg);  
end

delete(gcp('nocreate'));  % Close parallel pool
end
