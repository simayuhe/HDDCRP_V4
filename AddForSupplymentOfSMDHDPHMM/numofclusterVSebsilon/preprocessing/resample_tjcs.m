function tjc_res = resample_tjcs(save_dir,tjc_den,do_det_odd,do_plot,ds)
if ~exist(save_dir,'file')
    mkdir(save_dir);
end
if save_dir(end)~='\' && save_dir(end)~='/'
    save_dir = [save_dir '/'];
end
% ds = 1;
tjc_res = cell(length(tjc_den),1);
len_tjc = zeros(1,length(tjc_den));
if do_plot
    h = figure;
end
for ii = 1:length(tjc_den)
    x = tjc_den{ii};
    dx = x(2:end,:)-x(1:end-1,:);
    int = sqrt(sum(dx.*dx,2));
    int = mean(int);
    scale = 2 / int;
    x = scale*x;
    if do_det_odd
        y = tjc_preprocess3(x,ds,0,8);
    else
        y = tjc_preprocess(x,ds,0);
    end        
    y = y ./ scale;
    len_tjc(ii) = (size(y,1)-1)*ds/scale;
    tjc_res{ii} = y;
    if do_plot
        clf
        subplot(121)
        plot(tjc_den{ii}(:,1),tjc_den{ii}(:,2),'.-');
        subplot(122)
        plot(y(:,1),y(:,2),'.-');
        title(num2str(ii));
        input('Press enter to continue:');
        figure(h);
    end
end
save([save_dir 'tjc_res.mat'],'tjc_res','len_tjc');
