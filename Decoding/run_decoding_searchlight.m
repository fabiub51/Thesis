function run_decoding_searchlight(subj, design, smoothing)
    % This function runs the decoding analysis with specified subject, design, and smoothing using a whole-brain searchlight.
    subj = char(subj);
    design = char(design);
    %% Initialize config
    cfg = decoding_defaults;
    cfg.analysis = 'searchlight';  % Specify searchlight-based decoding (not ROI)
    cfg.decoding.software = 'libsvm';  % SVM library
    cfg.decoding.method = 'classification';

    % Base directory where all the data is stored
    basedir = '/Users/danieljanini/Documents/Thesis/miniblock/Outputs';

    % Define beta folder based on inputs
    beta_folder = fullfile(basedir, subj, strcat(smoothing, '_',subj,'_',design), 'nifti_betas');
    cfg.results.dir = fullfile(basedir, 'decoding', 'searchlight', design, subj);
    if ~exist(cfg.results.dir, 'dir')
        mkdir(cfg.results.dir);  % Create the directory if it doesn't exist
    end

    % Beta images
    n_betas = 240;
    beta_filenames = strcat('beta_', sprintfc('%04d', 1:n_betas), '.nii');
    cfg.files.name = fullfile(beta_folder, beta_filenames);
    cfg.files.name = cfg.files.name';  % Transpose to make it 240x1

    % No mask for searchlight (using whole brain)
    cfg.files.mask = '/Users/danieljanini/Documents/Thesis/Code/masking/group_mask.nii';

    cfg.results.output = {'accuracy_minus_chance'};

    %% Set labels
    pres_dir = '/Users/danieljanini/Documents/Thesis/Behavior/designmats';
    sub_index = subj(end-1:end);
    pattern = strcat('^P0',sub_index,'_.*_',design,'\.csv$');  % Filename pattern for the CSV files
    files = dir(fullfile(pres_dir, '*.csv'));
    filenames = {files.name};

    matches = ~cellfun('isempty', regexp(filenames, pattern));
    design_files = fullfile(pres_dir, filenames(matches));

    % Initialize labels array
    labels = [];
    for i = 1:length(design_files)
        data = readtable(design_files{i});
        [rows, cols] = size(data);
        for r = 1:rows
            for c = 1:cols
                if data{r,c} == 1
                    labels(end+1) = c;  % Condition number = column index
                end
            end
        end
    end

    cfg.files.label = labels(:);  % Ensure labels are a column vector

    %% Define chunks
    unique_labels = unique(labels);
    chunks = zeros(size(labels));
    
    for i = 1:length(unique_labels)
        condition_idx = find(labels == unique_labels(i));
        % Assign chunk numbers 1-6 to each repetition
        chunks(condition_idx) = 1:length(condition_idx);
    end
    
    cfg.files.chunk = chunks;
    %% Searchlight parameters
    cfg.searchlight.radius = 3;

    %% Decoding design
    cfg.design.function = 'make_design_cv';  % only set this, no call yet

    cfg.results.overwrite = 1;
    cfg.verbose = 1;
    cfg.scale.method = 'min0max1global'; 
    cfg.scale.estimation = 'all';
    cfg.design = make_design_cv(cfg); 
    %cfg.design.unbalanced_data = 'ok'; 
    cfg.design.fig = 0;   % after make_design_cv

    
    %% Run decoding
    decoding(cfg);  % Run the decoding analysis
end

