function [cluster stateCounts] = sample_c(data_struct,stateSeq,dist_struct,stateCounts)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Sample the cluster labels %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% alpha0 = hyperparams.alpha0_p_kappa0*(1-hyperparams.rho0);
% kappa0 = hyperparams.alpha0_p_kappa0*hyperparams.rho0;
% gamma0 = hyperparams.gamma0;
% beta_c = dist_struct.beta_c;

pi_z = dist_struct.pi_z;
pi_init = dist_struct.pi_init;
pi_c = dist_struct.pi_c;
Kc = size(pi_z,3);
Kz = size(pi_z,1);
L = zeros(1,Kc);
cluster = zeros(1,length(data_struct(1).test_cases));
for ii = data_struct(1).test_cases
    z = stateSeq(ii).z;    
    Nz = zeros(Kz,Kz);
    T = size(z,2);
    for tt = 2:T
        Nz(z(tt-1),z(tt)) = Nz(z(tt-1),z(tt)) + 1;
    end
    Nz = repmat(Nz,[1 1 Kc]);
    log_pi_c = log(pi_c+eps);
    log_pi_z = log(pi_z+eps);
    log_pi_init = log(pi_init+eps);
    Pc = log_pi_c + log_pi_init(z(1),:) + squeeze(sum(sum(Nz.*log_pi_z,1),2)).';
    Pc = exp(Pc-max(Pc));
    Pc = cumsum(Pc);
    c = 1 + sum(Pc(end)*rand > Pc);
    cluster(ii) = c;
    L(c) = L(c) + 1;
end

% for cc = 1:Kc
%     Pc(cc) = Pc(cc) + sum(dirln(pi_z(:,:,cc),alpha0*repmat(beta_c(:,cc).',Kz,1)+kappa0*eye(Kz)));
% end
% Pc = Pc + dirln(pi_init.',alpha0*beta_c.') + dirln(beta_c.',gamma0*beta_vec);

uniqueL = sum(L>0);
stateCounts.L = L;
stateCounts.uniqueL = uniqueL;

