function run_HDPHMM_inference(A_ALPHA,saveDir,trial_vec,codes,K,Kc,Kz,Ks,init_c,init_z,num_iter,resample_kappa,varargin)
	change_model = 1;
    sample_c = 1;
	if ~isempty(varargin)
		change_model = varargin{1};
		hyperparams = varargin{2};
		dist_struct = varargin{3};
		theta = varargin{4};
        sample_c = varargin{5};
	end

obsModelType = 'Multinomial';
priorType = 'DIR';
alpha = ones(1,K) ./ K;

%%
% Setting for inference:
clear settings

settings.Kc = Kc;
settings.Kz = Kz;   % truncation level for mode transition distributions
settings.Ks = Ks;  % truncation level for mode transition distributions
settings.Niter = num_iter;  % Number of iterations of the Gibbs sampler
settings.resample_kappa = resample_kappa;  % Whether or not to use sticky model
settings.sample_hypers = 1;
settings.saveMin = 100;
settings.seqSampleEvery = 100; % How often to run sequential z sampling
settings.saveEvery = 100;  % How often to save Gibbs sample stats
settings.storeEvery = 20;
settings.storeStateSeqEvery = 20;
settings.ploton = 0;  % Whether or not to plot the mode sequence while running sampler
settings.plotEvery = 20;
settings.plotpause = 0;  % Length of time to pause on the plot
settings.saveDir = saveDir;  % Directory to which to save files
settings.compute_likelihood = 1;
settings.formZInit = 1;
settings.formPInit = 0;
settings.change_model = change_model;
settings.sample_c = sample_c;
%%
% Set Hyperparameters

clear model

model.obsModel.type = obsModelType;

model.obsModel.priorType = priorType;

model.obsModel.params.alpha = alpha;

% Always using DP mixtures emissions, with single Gaussian forced by
% Ks=1...Need to fix.
model.obsModel.mixtureType = 'infinite';


% Sticky HDP-HMM parameter settings:
%model.HMMmodel.params.a_alpha=1;  % affects \pi_z%
model.HMMmodel.params.a_alpha=A_ALPHA;  % affects \pi_z
model.HMMmodel.params.b_alpha=0.01;
model.HMMmodel.params.a_gamma=1;  % global expected # of HMM states (affects \beta)
model.HMMmodel.params.b_gamma=0.01;
model.HMMmodel.params.a_gamma0 = 1;
model.HMMmodel.params.b_gamma0 = 0.01;
if settings.Kc>1
    model.HMMmodel.params.a_epsilon = 1;
    model.HMMmodel.params.b_epsilon = 0.01;
end
if settings.Ks>1
    model.HMMmodel.params.a_sigma = 1;
    model.HMMmodel.params.b_sigma = 0.01;
end
model.HMMmodel.params.c=1;  % self trans
model.HMMmodel.params.d=1;
model.HMMmodel.type = 'HDP';

J = length(codes);
data_struct = struct('obs',cell(1,J),'blockSize',cell(1,J));
data_struct(1).test_cases = 1:J;
for jj = 1:J
    data_struct(jj).obs = codes{jj};
    data_struct(jj).blockSize = ones(1,length(data_struct(jj).obs));
    data_struct(jj).c_init = init_c(jj);
    data_struct(jj).z_init = init_z{jj};
end
%%
for t=trial_vec

    settings.trial = t;  % Defines trial number, which is part of the filename used when saving stats
    if change_model
        HDPHMMDPinference(data_struct,model,settings);
    else
        HDPHMMDPinference(data_struct,model,settings,hyperparams,dist_struct,theta);
    end

end
