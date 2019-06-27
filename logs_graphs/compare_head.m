function fig = compare_head(log_date, str_type, R, C, head_id, save, filetype)
%COMPARE_HEAD This function plots the comparison (training vs. validation)
% for accuracy or loss of one specific head  in one graph
%   log_date = date of the log to use
%   str_type = "acc" for accuracy, "loss" for loss
%   (R,C) = starting coord of the data in .csv files
%   head_id = number of head to compare ("-1" for general loss)
%   save = if "true" saves the plots to file and auto-closes the figures
%   filetype = filetype of the saved plot ("pdf" or "png")

    % Fullscreen printing give problems with pdf saving
    if filetype == "pdf"
        fig = figure;
    else
        fig = figure('Units','normalized','Position',[0 0 1 1]);
    end
    
    hold on
    grid on
    grid minor
    
    if head_id == -1
        str_type_name = 'Loss';
    elseif str_type == "acc"
        str_type_name = 'Accuracy';
    elseif str_type == "loss"
        str_type_name = 'Loss';
    end
    
    if head_id == -1
        title(['Plot of ' ,str_type_name]);
    else
        title(['Plot of ' ,str_type_name, ' of head\_', num2str(head_id)]);
    end
    
    % read file
    if head_id == -1
        filename = sprintf([log_date, '/loss.csv'],head_id,str_type);
    else
        filename = sprintf([log_date, '/head_%d_%s.csv'],head_id,str_type);
    end
    tmp_array = csvread(filename,R,C);    
    
    % extract values
    % wall_time= tmp_array (:, 1);
    step = tmp_array (:, 2);
    value = tmp_array (:, 3);
    
    % read val file
    if head_id == -1
        filename = sprintf([log_date, '/val_loss.csv'],head_id,str_type);
    else
        filename = sprintf([log_date, '/val_head_%d_%s.csv'],head_id,str_type);
    end
    tmp_array = csvread(filename,R,C);
    % extract values
    % val_wall_time= tmp_array (:, 1);
    val_step = tmp_array (:, 2);
    val_value = tmp_array (:, 3);
    
    plot(step, value)
    plot(val_step, val_value)

    if head_id == -1
                legend(str_type_name, ['Validation ' , str_type_name],'Location', 'southoutside');
    else
        legend([str_type_name, ' of head\_', num2str(head_id)],...
               ['Validation ' , str_type_name, ' of head\_', num2str(head_id)],...
                'Location', 'southoutside');
    end
       
    if save
        if head_id == -1
            savename = sprintf([log_date, '/exported_img/compared_%s.%s'],str_type, filetype);
        else
            savename = sprintf([log_date, '/exported_img/compared_head_%d_%s.%s'],head_id,str_type, filetype);
        end
        saveas(fig,savename)
        close;
    end
    
end

