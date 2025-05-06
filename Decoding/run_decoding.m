function run_decoding(subj, design, smoothing, mask, output_dir)
    % This function runs the decoding analysis with specified subject, design, and smoothing.

    %% Initialize config
    cfg = decoding_defaults;
    cfg.analysis = 'roi';  % specify ROI-based decoding (not searchlight)
    cfg.decoding.software = 'libsvm';  % SVM library
    cfg.decoding.method = 'classification';

    % Base directory where all the data is stored
    basedir = '/Users/danieljanini/Documents/Thesis/miniblock/Outputs';

    % Ask user to specify the final part of the output folder
    output_subfolder = output_dir;

    % Define beta folder based on inputs
    beta_folder = fullfile(basedir, strcat(smoothing, '_',subj,'_',design), 'nifti_betas');
    cfg.results.dir = fullfile(basedir, 'decoding', 'ROI', output_subfolder, design, subj);

    % Beta images
    n_betas = 240;
    beta_filenames = strcat('beta_', sprintfc('%04d', 1:n_betas), '.nii');
    cfg.files.name = fullfile(beta_folder, beta_filenames);

    % Mask
    cfg.files.mask = mask;
    cfg.results.output = {'confusion_matrix'};

    %% Set labels
    pres_dir = '/Users/danieljanini/Documents/Thesis/Behavior/designmats';

    pattern = strcat('^P0',subj(end-1:end),'_.*_',design,'\.csv$');  % Filename pattern for the CSV files
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

    %% Decoding design
    cfg.design.function = 'make_design_cv';  % Define function to use for design creation
    cfg.design.label = 'leave_one_chunk_out';  % Cross-validation type
    cfg.design = make_design_cv(cfg);  % Create design matrix for cross-validation
    %cfg.design.unbalanced_data = 'ok';  % Allow unbalanced data (important for leave-one-chunk-out)
    cfg.results.overwrite = 1;  % Overwrite existing results
    cfg.design.fig = 0;
    %% Run decoding
    decoding(cfg);  % Run the decoding analysis
end
