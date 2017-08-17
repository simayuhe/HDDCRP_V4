function res_sub_tjc = resample_sub_tjcs(save_dir,tjc_pro,inds_sub_tjcs,varargin)
if ~exist(save_dir,'file')
	mkdir(save_dir);
end
if save_dir(end)~='\' && save_dir(end)~='/'
    save_dir = [save_dir '/'];
end
% clear;
% close all;
% load('tjc_pro.mat','tjc_pro');
% load('sub_tjc.mat', 'sub_tjc');
% img_bg = imread('background.bmp');
sum_sub_len = 0;
tot_num_sub = 0;
for ii = 1:length(inds_sub_tjcs)
    tot_num_sub = tot_num_sub + length(inds_sub_tjcs{ii});
    for jj = 1:length(inds_sub_tjcs{ii})
        sum_sub_len = sum_sub_len + length(inds_sub_tjcs{ii}{jj});
    end
end
ave_sub_len = round(sum_sub_len / tot_num_sub);
if ~isempty(varargin)
    ave_sub_len = varargin{1};
end
res_sub_tjc = zeros(tot_num_sub,ave_sub_len*2);
cnt = 0;
for ii = 1:length(inds_sub_tjcs)
    tjc_ii = tjc_pro{ii};
    for jj = 1:length(inds_sub_tjcs{ii})
        x0 = tjc_ii(inds_sub_tjcs{ii}{jj},:);
        x1 = tjc_preprocess2(x0,ave_sub_len);
        cnt = cnt+1;
        res_sub_tjc(cnt,1:ave_sub_len) = x1(:,1).';
        res_sub_tjc(cnt,ave_sub_len+1:end) = x1(:,2).';
    end
end
save([save_dir 'res_sub_tjc.mat'],'res_sub_tjc');
