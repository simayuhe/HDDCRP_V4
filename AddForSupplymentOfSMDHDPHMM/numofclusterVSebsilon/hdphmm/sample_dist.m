% function dist_struct = sample_dist(stateCounts,hyperparams,model)
% Sample transition distributions and mixture weights of MoG distributions
% given sampled hyperparameters and count matrices.

function dist_struct = sample_dist(stateCounts,hyperparams,model)

Kc = size(stateCounts.N,3);
Kz = size(stateCounts.Ns,1); % truncation level for transition distributions
Ks = size(stateCounts.Ns,2); % truncation level for mode-specific MoG emission distributions

% Define alpha0 and kappa0 in terms of alpha0+kappa0 and rho0:
alpha0 = hyperparams.alpha0_p_kappa0*(1-hyperparams.rho0);
kappa0 = hyperparams.alpha0_p_kappa0*hyperparams.rho0;
sigma0 = hyperparams.sigma0;
epsilon = hyperparams.epsilon;

N = stateCounts.N;  % N(i,j) = # z_t = i to z_{t+1}=j transitions. N(Kz+1,i) = 1 for i=z_1.
Ns = stateCounts.Ns;  % Ns(i,j) = # s_t = j given z_t=i
barM0 = stateCounts.barM0;  % barM(i,j) = # tables in restaurant i that considered dish j
L = stateCounts.L;
M = stateCounts.M;

switch model.HMMmodel.type
    case 'HDP'
        % Sample beta, the global menu, given new barM:
        gamma = hyperparams.gamma;
        beta_vec = randdirichlet((sum(M,2) + gamma/Kz))';
        gamma0 = hyperparams.gamma0;
        beta_c = zeros(Kz,Kc);
        for ii = 1:Kc
%             beta_c(:,ii) = randdirichlet((sum(barM0(:,:,ii),1))'+gamma0/Kz);
            beta_c(:,ii) = randdirichlet((sum(barM0(:,:,ii),1)+gamma0*beta_vec)');
        end
    case 'finite'
        % A finite HMM model with a sparse Dirichlet prior is exactly
        % equivalent to the truncated HDP-HMM model with a uniform global
        % menu:
        beta_vec = (1/Kz)*ones(1,Kz);
end

pi_z = zeros(Kz,Kz,Kc);
pi_s = zeros(Kz,Ks);
for ii = 1:Kc
    for jj=1:Kz
        kappa_vec = zeros(1,Kz);
        % Add an amount \kappa to Dirichlet parameter corresponding to a
        % self-transition:
        kappa_vec(jj) = kappa0;
        % Sample \pi_j's given sampled \beta_vec and counts N, where
        % DP(\alpha+\kappa,(\alpha\beta+\kappa\delta(j))/(\alpha+\kappa)) is
        % Dirichlet distributed over the finite partition defined by beta_vec:
        pi_z(jj,:,ii) = randdirichlet((alpha0*beta_c(:,ii)' + kappa_vec + N(jj,:,ii))')';
%         if sum(isnan(pi_z(jj,:,ii)))
%             a = 1;
%         end
    end
end
for jj = 1:Kz
    % Sample HMM-state-specific mixture weights \psi_j's with truncation
    % level Ks given sampled s stats Ns:
    pi_s(jj,:) = randdirichlet((Ns(jj,:) + sigma0/Ks)')';
end
pi_init = zeros(Kz,Kc);
for ii = 1:Kc
    pi_init(:,ii) = randdirichlet(alpha0*beta_c(:,ii) + N(Kz+1,:,ii)');
end
pi_c = randdirichlet((L+epsilon/Kc)')';

if isfield(stateCounts,'Nr')
    Nr = stateCounts.Nr;  % Nr(i) = # r_t = i
    Kr = length(Nr); % truncation level of measurement noise MoG
    eta0 = hyperparams.eta0;
    % Sample measurement noise mixture weights \pi_r with truncation
    % level Kr given sampled r stats Nr:
    dist_struct.pi_r = randdirichlet([Nr + eta0/Kr]')';
end

dist_struct.pi_z = pi_z;
dist_struct.pi_init = pi_init;
dist_struct.pi_s = pi_s;
dist_struct.pi_c = pi_c;
dist_struct.beta_c = beta_c;
dist_struct.beta_vec = beta_vec;
