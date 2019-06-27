close all
clear all
clc

%% Initialization

log_date = '2019_06_17_14_25';

%Print the figures (if false visualize it):
print = true;
filetype = 'pdf';

% Layout for "plot_compared_all" figure:
layout = [1,5];

% csvread('file.csv',R,C) 
% reads data from the comma separated value 
% formatted file starting at row and column (R,C).
R = 1;
C = 0; % 0 specifies the first value in the file.

%% Heads comparison plots

% Train
plot_together(log_date, 'acc',  R,C, false, print, filetype);
plot_together(log_date, 'loss', R,C, false, print, filetype);

% Validation
plot_together(log_date, 'acc',  R,C, true, print, filetype);
plot_together(log_date, 'loss', R,C, true, print, filetype);

%% Train-validation comparison plots

% Accuracy
for i = 0:4
    plot_compared(log_date, 'acc',  R,C, i, print, filetype);
end

% Loss
for i = 0:4
    plot_compared(log_date, 'loss',  R,C, i, print, filetype);
end

%% Train validation comparison (all together)

plot_compared_all(log_date, 'acc', R,C, print, filetype, layout);
plot_compared_all(log_date, 'loss',R,C, print, filetype, layout);

