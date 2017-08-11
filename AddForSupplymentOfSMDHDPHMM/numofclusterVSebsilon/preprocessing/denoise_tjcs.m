function tjc_den = denoise_tjcs(save_dir,tjc,do_med_flt,width,do_plot)
if ~exist(save_dir,'file')
    mkdir(save_dir);
end
if save_dir(end)~='\' && save_dir(end)~='/'
    save_dir = [save_dir '/'];
end
if do_plot
    h = figure;
end
tjc_den = cell(length(tjc),1);
for ii = 1:length(tjc)
    tjc_den{ii} = tjc_denoise2(tjc{ii});
%     tjc_den{ii} = tjc_denoise3(tjc{ii},3);
    if do_med_flt
        tjc_den{ii} = tjc_denoise4(tjc_den{ii},width);
    end
    if do_plot
        clf
        subplot(121)
        hold on
        plot(tjc{ii}(:,1),tjc{ii}(:,2),'.-');
        plot(tjc{ii}(end,1),tjc{ii}(end,2),'.-r');
%         title(['class' num2str(labels(ii))])
        subplot(122)
        hold on
        plot(tjc_den{ii}(:,1),tjc_den{ii}(:,2),'.-');
        plot(tjc_den{ii}(end,1),tjc_den{ii}(end,2),'.-r');
        title(['index' num2str(ii)]);
        input('press enter to continue:');
        figure(h);
    end
end
save([save_dir 'tjc_den.mat'],'tjc_den');
% plot_tjcs(tjc_den,1:length(tjc_den));