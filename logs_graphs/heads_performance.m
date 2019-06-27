function fig = heads_performance(log_date, str_type, R, C, is_val, save, filetype)
%HEADS_PERFORMANCE This function plots the accuracy or loss of all the
%heads in training or validation in one graph
%   log_date = date of the log to use
%   str_type = "acc" for accuracy, "loss" for loss
%   (R,C) = starting coord of the data in .csv files
%   is_val = "true" if for the validation, "false" if for the training 
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
    
    if str_type == "acc"
        str_type_name = 'Accuracy';
    elseif str_type == "loss"
        str_type_name = 'Loss';
    end

    
    for i = 0:4

        % read file
        if is_val
            filename = sprintf([log_date, '/val_head_%d_%s.csv'],i,str_type);
            title(['Plot of Validation ' str_type_name]);
        else 
            filename = sprintf([log_date, '/head_%d_%s.csv'],i,str_type);
            title(['Plot of ' str_type_name]);
        end
        tmp_array = csvread(filename,R,C);

        % extract values
        % wall_time= tmp_array (:, 1);
        step = tmp_array (:, 2);
        value = tmp_array (:, 3);

        if  i >= 1
            new_leg = [str_type_name, ' of head\_', num2str(i)]; % new legend
            h = [h; new_leg];
        else
            h = [str_type_name, ' of head\_', num2str(i)];
        end

        % plot in the main figure
        plot(step, value)

    end

    legend(h, 'Location', 'southoutside');
        
    if save
        savename = sprintf([log_date, '/exported_img/together_%s.%s'],str_type, filetype);
        saveas(fig,savename)
        close;
    end
    
end

