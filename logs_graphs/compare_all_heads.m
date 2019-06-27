function fig = compare_all_heads(log_date, str_type, R, C, save, filetype, layout)
%COMPARE_ALL_HEADS This function plots the comparison (training vs. validation)
% for accuracy or loss of all the heads in one graph
%   log_date = date of the log to use
%   str_type = "acc" for accuracy, "loss" for loss
%   (R,C) = starting coord of the data in .csv files
%   save = if "true" saves the plots to file and auto-closes the figures
%   filetype = filetype of the saved plot ("pdf" or "png")re

    % Fullscreen printing give problems with pdf saving
    fig = figure('Units','normalized','Position',[0 0 1 1]);
    
    for head_id = 0:4
        
        subplot(layout(1),layout(2), head_id+1)

        hold on
        grid on
        grid minor
        
        if str_type == "acc"
            str_type_name = 'Accuracy';
        elseif str_type == "loss"
            str_type_name = 'Loss';
        end

        title(['Plot of ' str_type_name, ' of head\_', num2str(head_id)]);

        % read file
        filename = sprintf([log_date, '/head_%d_%s.csv'],head_id,str_type);
        tmp_array = csvread(filename,R,C);
        % extract values
        % wall_time= tmp_array (:, 1);
        step = tmp_array (:, 2);
        value = tmp_array (:, 3);

        % read val file
        filename = sprintf([log_date, '/val_head_%d_%s.csv'],head_id,str_type);
        tmp_array = csvread(filename,R,C);
        % extract values
        % val_wall_time= tmp_array (:, 1);
        val_step = tmp_array (:, 2);
        val_value = tmp_array (:, 3);

        plot(step, value)
        plot(val_step, val_value)

        legend([str_type_name, ' of head\_', num2str(head_id)],...
               ['Validation ' , str_type_name, ' of head\_', num2str(head_id)],...
                'Location', 'southoutside');
            
    end

    if save
        savename = sprintf([log_date, '/exported_img/compared_head_all_%s.%s'],str_type, filetype);
        saveas(fig,savename)
        close;
    end
    
end

