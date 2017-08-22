
clear;
close all;

addpath_script

% result_dir = './results/CASIA2/';
% raw_data_file = './data/CASIA/CASIA_tjc_small.mat';
result_dir = './results_1000/CASIA/';
raw_data_file = './data/CASIA/CASIA_tjc.mat';%ԭʼ��ݼ�
myiter=[1 10 100 1000];
%myiter=[1 500 1000 1500 2000 2500 3000];
%matlabpool('open',4);
for myi=1:1:length(myiter)
    for myj=1:3
        start_exp = myiter(myi)+myj;
        num_labs = 1;
        %��ͷ���?�����˲������ʱ��������Լ��켣�����̶�����������д�����
        
        do_pre_process = 0;%ѡ���Ƿ����֮ǰ�Ĵ��?�˽�һ����Щ��Ҫÿ�ζ�������Щ����һ�ξ͹���
        do_denoise = 0;
        do_med_flt = 0;
        med_width = 5;
        do_resample = 0;
        do_plot_denoise = 0;
        do_det_odd = 0;
        ds = 1;
        
        do_encode = 0;
        do_divide = 1;
        ave_num_seg = 8;
        do_plot_divide = 0;
        do_resample_subs = 1;
        do_gen_features = 1;
        do_sc = 1;
        n_dim = 10;
        K = 200;
        
        do_compute_D = 0;
        
        if do_pre_process
            load(raw_data_file,'tjc');
            pre_process_tjcs(result_dir,tjc,do_denoise,do_med_flt,med_width,...
                do_resample,do_plot_denoise,do_det_odd,ds);
        end
        
        % compute distance matrix
        if do_compute_D
            load([result_dir 'tjc_res.mat'], 'tjc_res');
            D = compute_DTW_matrix(tjc_res);
            save([result_dir 'dist_res_DTW.mat'],'D');
        end
        
        if do_encode
            load([result_dir 'tjc_res.mat'],'tjc_res','len_tjc');
            code_trajectories(result_dir,tjc_res,len_tjc,do_divide,ave_num_seg,do_plot_divide,...
                do_resample_subs,do_gen_features,n_dim,K,do_sc);
        end
        
        Kz = 60;%*ones(1,num_labs);
        Kc = 40;%*ones(1,num_labs);%�ĵ���ĳ�ʼ����
        Ks = 1;%*ones(1,num_labs);
        %matlabpool('open', num_labs);
        % Kz = Composite();
        % Kc = Composite();
        % Ks = Composite();
        % exp_num = Composite();
        % save_dir = Composite();
        % codes = Composite();
        % K = Composite();
        % init_z_temp = Composite();
        % init_c_temp = Composite();
        S_feat = load([result_dir 'features.mat'],'features');
        S_inds = load([result_dir 'inds_sub_tjcs'], 'inds_sub_tjcs');
        S_D = load([result_dir 'dist_res_DTW.mat'],'D');
        S_code = load([result_dir 'code.mat'],'codes','K');
        % for ii = 1:num_labs
        % 	Kz{ii} = Kz_vec(ii);
        % 	Kc{ii} = Kc_vec(ii);
        % 	Ks{ii} = Ks_vec(ii);
        % 	exp_num{ii} = ii+start_exp;
        exp_num=start_exp;
        % 	save_dir{ii} = [result_dir 'HMMdata/exp' num2str(exp_num{ii}) '/'];
        save_dir = [result_dir 'HMMdata/exp' num2str(exp_num) '/'];
        if ~exist(save_dir,'file')
            mkdir(save_dir);
        end
        codes = S_code.codes;
        K = S_code.K;
        init_z = encode_tjcs(S_feat.features,S_inds.inds_sub_tjcs,Kz,do_sc);
        init_z_temp = init_z;
        init_c = compute_init_c(S_D.D, Kc);
        init_c_temp = init_c;
        save([save_dir 'init_z.mat'], 'init_z');
        save([save_dir 'init_c.mat'], 'init_c');
        % end
        
        init_z = init_z_temp;
        init_c = init_c_temp;
        clear init_z_temp init_c_temp S_feat S_inds S_D S_code
        
        
        trial_vec = 1;
        resample_kappa = 1;
        num_iter = 3000;
        A_EBSILON=myiter(myi);
        run_HDPHMM_inference(A_EBSILON,save_dir,trial_vec,codes,K,Kc,Kz,Ks,init_c,init_z,num_iter,resample_kappa);
        
        
        %%  
        %��¼�������
        
        nker=start_exp;
        filename=strcat([result_dir 'HMMdata/exp' num2str(nker) '/'],'/HDPHMMDPstats','iter',num2str(1000),'trial',num2str(1));
        load ([filename,'.mat'])
        NumberC=0;
        for ii=1:1:40
            if (size(find(S.cluster(1,:)==ii),2)==0)
                continue;
            else
                NumberC=NumberC+1;
            end
        end
        fprintf(1,'Number of clusters is %d .\n',NumberC);
        
        save([save_dir 'NumberC.mat'],'NumberC');
    end
    
end
%matlabpool close;