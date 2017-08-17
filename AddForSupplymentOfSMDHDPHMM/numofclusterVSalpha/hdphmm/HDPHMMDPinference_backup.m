%%%%%%%%%%%%%%%%%%%%%%%%%%% HDPHMMDPinference.m %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%%%           ****SEE 'utilities/runstuff.m' FOR EXAMPLE INPUTS *****
%%
%% Inputs:
%%%% data_struct - structure of observations, initial segmentation of data, etc.
%%%% model - structure containing hyperparameters for transition distributions and dynamic parameters 
%%%% settings - structure of settings including number of Gibbs iterations, directory to save statistics to, how often to save, etc.
%%
%% Outputs
%%%% various statistics saved at preset frequency to
%%%% settings.saveDir/HDPHMMDPstatsiter[insert iter #]trial[insert trial #].mat
%%%% in a structure of the form S(store_count).field(time_series).subfield
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function HDPHMMDPinference(data_struct,model,settings,restart)

trial = settings.trial;
if ~isfield(settings,'saveMin')
    settings.saveMin = 1;
end
resample_kappa = settings.resample_kappa;
Kz = settings.Kz;
Niter = settings.Niter;

display(strcat('Trial:',num2str(trial)))

%%%%%%%%%% Generate observations (if not provided) %%%%%%%%%%
%%%                       and                             %%%
%%%%%%%%%%        Initialize variables             %%%%%%%%%%

% Set the starting iteration:
n_start = 1;

% Build initial structures for parameters and sufficient statistics:
[theta Ustats stateCounts stateSeq INDS hyperparams data_struct model S] = initializeStructs(model,data_struct,settings);

obsModel = model.obsModel;  % structure containing the observation model parameters
obsModelType = obsModel.type;   % type of emissions including Gaussian, multinomial, AR, and SLDS.
HMMhyperparams = model.HMMmodel.params; % hyperparameter structure for the HMM parameters
HMMmodelType = model.HMMmodel.type; % type of HMM including finite and HDP

% Resample concentration parameters:
hyperparams = sample_hyperparams_init(stateCounts,hyperparams,HMMhyperparams,HMMmodelType,resample_kappa);

% Sample the transition distributions pi_z, initial distribution
% pi_init, emission weights pi_s, and global transition distribution beta
% (only if HDP-HMM) from the priors on these distributions:
dist_struct = sample_dist(stateCounts,hyperparams,model);

% If the optional 'formZInit' option has been added to the settings
% structure, then form an initial mode sequence in one of two ways.  If
% 'z_init' is a field of data_struct, then the specified initial
% sequence will be used. Otherwise, the sequence will be sampled from
% the prior.
if isfield(settings,'formZInit')
    if settings.formZInit == 1
        [cluster stateSeq INDS stateCounts] = sample_czs_init(data_struct,dist_struct,obsModelType);
        stateCounts = sample_tables(stateCounts,hyperparams,dist_struct.beta_c,dist_struct.beta_vec,Kz);
        dist_struct = sample_dist(stateCounts,hyperparams,model);
        hyperparams = sample_hyperparams(stateCounts,hyperparams,HMMhyperparams,HMMmodelType,resample_kappa);
        display('Forming initial z using specified z_init or sampling from the prior using whatever fixed data is available')
    end
elseif length(data_struct)>length(data_struct(1).test_cases)
    display('Do you want z_init set to truth for extra datasets?  If so, make setttings.formZinit =1 ')
end

if settings.formPInit
    theta = init_theta(theta, dist_struct);
else
    % Sample emission params theta_{z,s}'s. If the above 'formZInit' option
    % was not utilized, the initial parameters will just be drawn from the
    % prior.
    if settings.formZInit == 1
        Ustats = update_Ustats(data_struct,INDS,stateCounts,obsModelType);
    end
    theta = sample_theta(theta,Ustats,obsModel);
end

% Create directory in which to save files if it does not currently
% exist:
if ~exist(settings.saveDir,'file')
    mkdir(settings.saveDir);
end

% Save initial statistics and settings for this trial:
if isfield(settings,'filename')
    settings_filename = strcat(settings.saveDir,'/',settings.filename,'_info4trial',num2str(trial));    % create filename for current iteration
    init_stats_filename = strcat(settings.saveDir,'/',settings.filename,'initialStats_trial',num2str(trial));    % create filename for current iteration
else
    settings_filename = strcat(settings.saveDir,'info4trial',num2str(trial));    % create filename for current iteration
    init_stats_filename = strcat(settings.saveDir,'initialStats_trial',num2str(trial));    % create filename for current iteration
end
save(settings_filename,'data_struct','settings','model') % save current statistics
save(init_stats_filename,'dist_struct','theta','hyperparams') % save current statistics
    
% If the 'ploton' option is included in the settings structure (and if it
% is set to 1), then create a figure for the plots:
if isfield(settings,'ploton')
    if settings.ploton == 1
        H = figure;
    end
end

%%%%%%%%%% Run Sampler %%%%%%%%%%
tic
for n=n_start:Niter
    fprintf(1,'iteration %d\n',n);
    % Sample z and s sequences given data, transition distributions,
    % HMM-state-specific mixture weights, and emission parameters:
    % Block sample (z_{1:T},s_{1:T})|y_{1:T}
    [stateSeq INDS stateCounts] = sample_zs(cluster,data_struct,dist_struct,theta,stateCounts,stateSeq,INDS,obsModelType);
 
    % Based on mode sequence assignment, sample how many tables in each
    % restaurant are serving each of the selected dishes. Also sample the
    % dish override variables:
    stateCounts = sample_tables(stateCounts,hyperparams,dist_struct.beta_c,dist_struct.beta_vec,Kz);
    
    % Sample the transition distributions pi_z, initial distribution
    % pi_init, emission weights pi_s, and avg transition distribution beta:
    dist_struct = sample_dist(stateCounts,hyperparams,model);
    
    % Create sufficient statistics:
    Ustats = update_Ustats(data_struct,INDS,stateCounts,obsModelType);
    % Sample theta_{z,s}'s conditioned on the z and s sequences and the
    % sufficient statistics structure Ustats:
    theta = sample_theta(theta,Ustats,obsModel);

    % Sample c
    [cluster stateCounts] = sample_c(stateSeq,dist_struct,stateCounts);

    % Resample concentration parameters:
    if settings.sample_hypers
        hyperparams = sample_hyperparams(stateCounts,hyperparams,HMMhyperparams,HMMmodelType,resample_kappa);
    end
    
    % Compute likelihood
    total_log_likelihood = 0;
    if settings.compute_likelihood
        if n > settings.saveMin - settings.saveEvery
            for ii = 1:length(data_struct)
                [neglog_c,~,~] = observation_likelihood(data_struct(ii),obsModelType,dist_struct,theta);
                pi_c = dist_struct.pi_c;
                total_log_likelihood = total_log_likelihood + log(eps+pi_c*exp(neglog_c)');
%                 if(isnan(total_log_likelihood))
%                     a = 1;
%                 end
            end
        end
    end
    
    % Build and save stats structure:
    S = store_stats(S,n,settings,cluster,stateSeq,dist_struct,theta,hyperparams,stateCounts,total_log_likelihood);
    toc
    fprintf(1,'Average time is %f seconds.\n',toc/n);
    fprintf(1,'Estimated time is %f hours.\n',toc/n*Niter/3600);
end
