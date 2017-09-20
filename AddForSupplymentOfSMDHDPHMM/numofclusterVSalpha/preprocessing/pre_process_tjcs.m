function pre_process_tjcs(save_dir,tjc,do_denoise,do_med_flt,med_width,do_resample,do_plot,do_det_odd,ds)
if ~exist(save_dir,'file')
    mkdir(save_dir);
end
if save_dir(end)~='\' && save_dir(end)~='/'
    save_dir = [save_dir '/'];
end
if do_denoise
    tjc_den = denoise_tjcs(save_dir,tjc,do_med_flt,med_width,do_plot);
else
    load([save_dir 'tjc_den.mat'],'tjc_den');
end

if do_resample
    tjc_res = resample_tjcs(save_dir,tjc_den,do_det_odd,do_plot,ds);
else
    load([save_dir 'tjc_res.mat'],'tjc_res','len_tjc');
end
