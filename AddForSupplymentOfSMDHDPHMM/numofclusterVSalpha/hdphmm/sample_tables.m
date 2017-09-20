% function stateCounts = sample_tables(stateCounts,hyperparams,beta_vec,Kz)
% Sample the number of tables in restaurant i serving dish j given the mode
% sequence z_{1:T} and hyperparameters. Also sample the override variables.

function stateCounts = sample_tables(stateCounts,hyperparams,beta_c,beta_vec,Kz)
   
% Split \alpha and \kappa using \rho:
rho0 = hyperparams.rho0;
alpha0 = hyperparams.alpha0_p_kappa0*(1-rho0);
kappa0 = hyperparams.alpha0_p_kappa0*rho0;
gamma0 = hyperparams.gamma0;

N = stateCounts.N;

Kc = size(beta_c,2);
M0 = zeros(Kz+1,Kz,Kc);
barM0 = zeros(Kz+1,Kz,Kc);
sum_w = zeros(Kz,Kc);
for cc = 1:Kc
    M0(:,:,cc) = randnumtable([alpha0*repmat(beta_c(:,cc)',Kz,1)+kappa0*eye(Kz); alpha0*beta_c(:,cc)'],N(:,:,cc));
    [barM0(:,:,cc) sum_w(:,cc)] = sample_barM(M0(:,:,cc),beta_c(:,cc)',rho0);
end
M = randnumtable(gamma0*repmat(beta_vec',1,Kc),squeeze(sum(barM0,1)));

stateCounts.M0 = M0;
stateCounts.barM0 = barM0;
stateCounts.sum_w = sum_w;
stateCounts.M = M;