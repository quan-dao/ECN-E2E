function plot_log_graphs(log_date, print, filetype, layout)
%PLOT_LOG_GRAPHS This function plots/prints all the necessary data from the
% tensorboard callback .csv extracted files
%   log_date = date of the log to use
%   print = if "true" saves the plots to file and auto-closes the figures
%   filetype = filetype of the saved plot ("pdf" or "png")
%   layout = Layout for "plot_compared_all" figure (e.g. [1;5];)
%
%   IMPORTANT: create the  log_dir/exported_img/ folder!!
%
%   Example:
%   plot_log_graphs('2019_06_25_13_30', true, 'pdf', [1;5])

    %% Initialization

    % csvread('file.csv',R,C) 
    % reads data from the comma separated value 
    % formatted file starting at row and column (R,C).
    R = 1;
    C = 0; % 0 specifies the first value in the file.

    %% Heads comparison plots

    % Train
    heads_performance(log_date, 'acc',  R,C, false, print, filetype);
    heads_performance(log_date, 'loss', R,C, false, print, filetype);

    % Validation
    heads_performance(log_date, 'acc',  R,C, true, print, filetype);
    heads_performance(log_date, 'loss', R,C, true, print, filetype);

    %% Train-validation comparison plots

    % Accuracy
    for i = 0:4
        compare_head(log_date, 'acc',  R,C, i, print, filetype);
    end

    % Loss
    for i = 0:4
        compare_head(log_date, 'loss',  R,C, i, print, filetype);
    end

    %% Train validation comparison (all together)

    compare_all_heads(log_date, 'acc', R,C, print, filetype, layout);
    compare_all_heads(log_date, 'loss',R,C, print, filetype, layout);

end