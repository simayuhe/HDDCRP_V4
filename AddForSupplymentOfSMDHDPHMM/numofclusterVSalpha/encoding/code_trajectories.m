function code_trajectories(save_dir,tjc_res,len_tjc,do_divide,ave_num_seg,do_plot,...
		do_resample_subs,do_gen_features,n_dim,K,do_sc,varargin)
    is_test = 0;
if ~isempty(varargin)
    ave_len_seg = varargin{1};
    ave_sub_len = varargin{2};
    COEFF = varargin{3};
    M = varargin{4};
    centers = varargin{5};
    is_test = 1;
end
if ~exist(save_dir,'file')
	mkdir(save_dir);
end
if save_dir(end)~='\' && save_dir(end)~='/'
    save_dir = [save_dir '/'];
end

if do_divide
    if ~is_test
        inds_sub_tjcs = divide_tjcs(save_dir,tjc_res,len_tjc,do_plot,ave_num_seg);
    else
        inds_sub_tjcs = divide_tjcs(save_dir,tjc_res,len_tjc,do_plot,ave_num_seg,ave_len_seg);
    end
else
    load([save_dir 'inds_sub_tjcs.mat'],'inds_sub_tjcs');
end

if do_resample_subs
    if ~is_test
    res_sub_tjc = resample_sub_tjcs(save_dir,tjc_res,inds_sub_tjcs);
    else
        res_sub_tjc = resample_sub_tjcs(save_dir,tjc_res,inds_sub_tjcs,ave_sub_len);
    end
else
    load([save_dir 'res_sub_tjc.mat'],'res_sub_tjc');
end

if do_gen_features
    if ~is_test
    features = generate_feats(save_dir,res_sub_tjc,n_dim);
    else
        features = generate_feats(save_dir,res_sub_tjc,n_dim,COEFF,M);
    end
else
    load([save_dir 'features.mat'],'features');
end

% Encode tjcs

if ~is_test
codes = encode_tjcs(features,inds_sub_tjcs,K,do_sc);
else
    codes = encode_tjcs(features,inds_sub_tjcs,K,do_sc,centers);
end
save([save_dir 'code.mat'],'codes','K');

