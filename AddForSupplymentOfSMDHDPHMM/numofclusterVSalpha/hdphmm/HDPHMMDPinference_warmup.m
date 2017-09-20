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

function HDPHMMDPinference(data_struct,model,settings,varargin)

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

if isfield(settings,'restart')
    % The optional 'restart' argument allows you to restart the inference
    % algorithm using the statistics that were stored at settings.lastSave
    restart = settings.restart;
    if restart==1
        n = settings.lastSave;
        
        % Build necessary structures and clear structures that exist as
        % part of the saved statistics:
	[theta Ustats stateCounts stateSeq INDS hyperparams data_struct model S] = initializeStructs(model,data_struct,settings);
        clear theta Ustats hyperparams S
        
        % Load the last saved statistics structure S:
        if isfield(settings,'filename')
            filename = strcat(settings.saveDir,'/',settings.filename,'iter',num2str(n),'trial',num2str(settings.trial));    % create filename for current iteration
        else
            filename = strcat(settings.saveDir,'/HDPHMMDPstats','iter',num2str(n),'trial',num2str(settings.trial));    % create filename for current iteration
        end
        
        load(filename)
        
        obsModel = model.obsModel;  % structure containing the observation model parameters
        obsModelType = obsModel.type;   % type of emissions including Gaussian, multinomial, AR, and SLDS.
        HMMhyperparams = model.HMMmodel.params; % hyperparameter structure for the HMM parameters
        HMMmodelType = model.HMMmodel.type; % type of HMM including finite and HDP
        
        % Grab out the last saved statistics from the S structure:
        numSaves = settings.saveEvery/settings.storeEvery;
        numStateSeqSaves = settings.saveEvery/settings.storeStateSeqEvery;
        theta = S.theta(numSaves);
        dist_struct = S.dist_struct(numSaves);
        hyperparams = S.hyperparams(numSaves);
        stateSeq = S.stateSeq(numStateSeqSaves);
        cluster = S.cluster(numSaves,:);
        
        [~, ~, ~, ~, ~, ~, ~, ~, S] = initializeStructs(model,data_struct,settings);
        % Set new save counter variables to 1:
        S.m = 1;
        S.n = 1;
        
        % Set the new starting iteration to be lastSave + 1:
        n_start = n + 1;
        
    end
else
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
            if isempty(varargin)
                dist_struct = sample_dist(stateCounts,hyperparams,model);
                hyperparams = sample_hyperparams(stateCounts,hyperparams,HMMhyperparams,HMMmodelType,resample_kappa);
            else
                hyperparams = varargin{1};
                dist_struct = varargin{2};
            end
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
        if isempty(varargin)
            theta = sample_theta(theta,Ustats,obsModel);
        else
            theta = varargin{3};
        end
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
end
    
% If the 'ploton' option is included in the settings structure (and if it
% is set to 1), then create a figure for the plots:
if isfield(settings,'ploton')
    if settings.ploton == 1
        H = figure;
    end
end

%%%%%%%%%% Run Sampler %%%%%%%%%%
warmup = settings.warmup;
tic
for n=n_start:Niter
    fprintf(1,'iteration %d\n',n);
    % Sample z and s sequences given data, transition distributions,
    % HMM-state-specific mixture weights, and emission parameters:
    % Block sample (z_{1:T},s_{1:T})|y_{1:T}
%     tic
    [stateSeq INDS stateCounts] = sample_zs(cluster,data_struct,dist_struct,theta,stateCounts,stateSeq,INDS,obsModelType);
%     fprintf(1,'sample_zs: ');
%     toc
	if settings.change_model
		% Based on mode sequence assignment, sample how many tables in each
		% restaurant are serving each of the selected dishes. Also sample the
		% dish override variables:
%     tic
		stateCounts = sample_tables(stateCounts,hyperparams,dist_struct.beta_c,dist_struct.beta_vec,Kz);
%     fprintf(1,'sample_tables: ');
%         toc
		
		% Sample the transition distributions pi_z, initial distribution
		% pi_init, emission weights pi_s, and avg transition distribution beta:
%     tic
		dist_struct = sample_dist(stateCounts,hyperparams,model);
%     fprintf(1,'sample_dist: ');
%         toc
		
		% Create sufficient statistics:
%     tic
		Ustats = update_Ustats(data_struct,INDS,stateCounts,obsModelType);
%     fprintf(1,'update_Ustats: ');
%         toc

        
		% Sample theta_{z,s}'s conditioned on the z and s sequences and the
		% sufficient statistics structure Ustats:
%     tic
		theta = sample_theta(theta,Ustats,obsModel);
%     fprintf(1,'sample_theta: ');
%     toc
        
	end

    % Sample c
    if settings.sample_c && n > warmup
%     tic
        [cluster stateCounts] = sample_c(data_struct,stateSeq,dist_struct,stateCounts);
%     fprintf(1,'sample_c: ');
%     toc
    end

    % Resample concentration parameters:
    if settings.change_model
    if settings.sample_hypers
%     tic
        hyperparams = sample_hyperparams(stateCounts,hyperparams,HMMhyperparams,HMMmodelType,resample_kappa);
%     fprintf(1,'sample_hyper: ');
%     toc
    end
    end
    
    % Compute likelihood
    total_log_likelihood = 0;
    if settings.compute_likelihood
%         tic
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
%         fprintf(1,'compute_likelihood: ');
%         toc
    end
    
    % Build and save stats structure:
%     tic
    S = store_stats(S,n,settings,cluster,stateSeq,dist_struct,theta,hyperparams,stateCounts,total_log_likelihood);
%     fprintf(1,'store_stats: ');
%     toc
    
    toc
    fprintf(1,'Average time is %f seconds.\n',toc/n);
    fprintf(1,'Estimated time is %f hours.\n',toc/n*Niter/3600);
end
