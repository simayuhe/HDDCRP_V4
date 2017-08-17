% function [hyperparams] = sample_hyperparams(stateCounts,hyperparams,hyperhyperparams,HMMmodelType,resample_kappa)
% Sample concentration parameters that define the distribution on transition
% distributions and mixture weights of the various model components.

function [hyperparams] = sample_hyperparams(stateCounts,hyperparams,hyperhyperparams,HMMmodelType,resample_kappa)

% Hyperparams for Gamma dist over \alpha+\kappa, where transition distributions
% \pi_j \sim DP(\alpha+\kappa,(\alpha\beta+\kappa\delta(j))/(\alpha+\kappa))
% = DP(alpha_p_kappa, (1-\rho)*\beta + \rho*\delta(j)):
a_alpha=hyperhyperparams.a_alpha;
b_alpha=hyperhyperparams.b_alpha;

% Hyperparams for Beta dist over \rho, where \rho relates \alpha+\kappa to
% \alpha and \kappa individually.
c=hyperhyperparams.c;
d=hyperhyperparams.d;

% Grab out last value of the hyperparameters:
alpha0_p_kappa0 = hyperparams.alpha0_p_kappa0;
sigma0 = hyperparams.sigma0;
epsilon = hyperparams.epsilon;

L = stateCounts.L;
uniqueL = stateCounts.uniqueL;
N = stateCounts.N; % N(i,j) = # z_t = i to z_{t+1}=j transitions in z_{1:T}. N(Kz+1,i) = 1 for i=z_1.
Ns = stateCounts.Ns; % Ns(i,k) = # of obs assigned to mix component k in mode i (i.e., # s_t = k given z_t=i)
uniqueS = stateCounts.uniqueS; % uniqueS(i) = sum_j Ns(i,j) = # of mixture components for HMM-state i
M = stateCounts.M; % M(i,j) = # of tables in restaurant i serving dish k
M0 = stateCounts.M0;
barM0 = stateCounts.barM0; % barM(i,j) = # of tables in restaurant i considering dish k
sum_w = stateCounts.sum_w; % sum_w(i) = # of overriden dish assignments in restaurant i

Nkdot = sum(N,2);
M0kdot = sum(M0,2);
barM0cdot = sum(sum(barM0,1),2);
Mcdot = sum(M,1);
barK = sum(sum(M,2)>0);
Nskdot = sum(Ns,2);

validindices = find(Nkdot>0);
validindices2 = find(Nskdot>0);

switch HMMmodelType

    case 'HDP'

        % Hyperparams for gamma dist over \gamma, where avg transition distribution
        % \beta \sim stick(\gamma):
        a_gamma=hyperhyperparams.a_gamma;
        b_gamma=hyperhyperparams.b_gamma;
        
        a_gamma0=hyperhyperparams.a_gamma0;
        b_gamma0=hyperhyperparams.b_gamma0;

        gamma = hyperparams.gamma;
        gamma0 = hyperparams.gamma0;

        % Resample concentration parameters:
        if isempty(validindices)
            alpha0_p_kappa0 = randgamma(a_alpha) / b_alpha;    % Gj concentration parameter
            gamma = randgamma(a_gamma) / b_gamma;
            gamma0 = randgamma(a_gamma0) / b_gamma0;    % G0 concentration parameter
        else
            alpha0_p_kappa0  = gibbs_conparam(alpha0_p_kappa0, Nkdot(validindices),M0kdot(validindices),a_alpha,b_alpha,50);
            gamma = gibbs_conparam(gamma,sum(Mcdot),barK,a_gamma,b_gamma,50);
            validindices3 = find(barM0cdot>0);
            gamma0 = gibbs_conparam(gamma0,barM0cdot(validindices3),Mcdot(validindices3),a_gamma0,b_gamma0,50);
        end
        

    case 'finite'

        % Resample Dirichlet parameter for \pi_j \sim
        % Dir(\alpha/L,...,\alpha/L + \kappa,...,\alpha/L):
        if isempty(validindices)
            alpha0_p_kappa0 = randgamma(a_alpha) / b_alpha;    
        else
            alpha0_p_kappa0  = gibbs_conparam(alpha0_p_kappa0, Nkdot(validindices),Mkdot(validindices),a_alpha,b_alpha,50);
        end
end

if size(Ns,2)>1 % Only spend time resampling sigma0 if MoG mode-specific emission distribution

    % Hyperparams for Gamma dist over \sigma, where HMM-state-specific mixture
    % weights \psi_j \sim stick(\sigma):
    a_sigma=hyperhyperparams.a_sigma;
    b_sigma=hyperhyperparams.b_sigma;

    if isempty(validindices2)
        sigma0 = randgamma(a_sigma) / b_sigma;
    else
        sigma0 = gibbs_conparam(sigma0,Nskdot(validindices2),uniqueS(validindices2),a_sigma,b_sigma,50);
    end

else
    sigma0 = 1;
end

a_epsilon = hyperhyperparams.a_epsilon;
b_epsilon = hyperhyperparams.b_epsilon;
epsilon = gibbs_conparam(epsilon,sum(L),uniqueL,a_epsilon,b_epsilon,50);

if resample_kappa  % Only spend time resampling rho0 if sticky model
    
    % Resample self-transition proportion parameter:
    total_w = sum(sum(sum_w));
    rho0 = betarnd(c+total_w,d+(sum(sum(sum(M0)))-total_w));
else
    rho0 = 0; %betarnd(0,1);
end

if isfield(stateCounts,'Nr')
    
    if length(stateCounts.Nr)>1  % Only spend time resampling eta0 if MoG measurement noise
    
        Nr = stateCounts.Nr;  % Nr(i) = number of observations assigned to the ith mix component of MoG measurement noise model
        eta0 = hyperparams.eta0;
        
        % Hyperparams for Gamma dist over \eta, where measurement noise mixture
        % weights \pi_r \sim stick(\eta):
        a_eta=hyperhyperparams.a_eta;
        b_eta=hyperhyperparams.b_eta;
        
        num_unique_r = length(find(Nr)); % number of unique mix components used
        T = sum(Nr);  % number of observations
        
        % Resample concentration parameter:
        if num_unique_r==0
            eta0 = randgamma(a_eta) / b_eta;
        else
            eta0 = gibbs_conparam(eta0,T,num_unique_r,a_eta,b_eta,50);
        end
    else
        eta0 = 1;
    end
        
    hyperparams.eta0 = eta0;
    
end

hyperparams.gamma = gamma;
hyperparams.gamma0 = gamma0;
hyperparams.alpha0_p_kappa0 = alpha0_p_kappa0;
hyperparams.sigma0 = sigma0;
hyperparams.rho0 = rho0;
hyperparams.epsilon = epsilon;
