function oz_lik = oz_likelihood(likelihood,loglike_normalizer,blockEnd,z,pi_z,pi_s,pi_init)
% This funtion need improve

% Allocate storage space
Kz = size(pi_z,2);
Ks = size(pi_s,2);
T  = length(blockEnd);

% fwd_msg     = ones(Kz,T);
% partial_marg = zeros(Kz,T);
% neglog_c = zeros(1,T);

% Compute marginalized likelihoods for all times, integrating s_t
% [~,inds] = max(likelihood .* pi_s(:,:,ones(1,1,blockEnd(end))),[],2);
% marg_like = reshape(likelihood(inds),Kz,blockEnd(end));

marg_like = squeeze(sum(likelihood .* pi_s(:,:,ones(1,1,blockEnd(end))),2));
% marg_like = squeeze(sum(likelihood .* pi_s(:,:,ones(1,1,blockEnd(end))),2));

% If necessary, combine likelihoods within blocks, avoiding underflow
if T < blockEnd(end)
  marg_like = log(marg_like+eps);

  log_block_like = zeros(Kz,T);
  log_block_like(:,1) = sum(marg_like(:,1:blockEnd(1)),2);
  % Initialize normalization constant to be that due to the likelihood:
  for tt = 2:T
    log_block_like(:,tt) = sum(marg_like(:,blockEnd(tt-1)+1:blockEnd(tt)),2);
  end

%   block_norm = max(log_block_like,[],1);
%   log_block_like = exp(log_block_like - block_norm(ones(Kz,1),:));
%   % Add on the normalization constant used after marginalizing the s_t's:
%   neglog_c = neglog_c + block_norm;
else
    
  log_block_like = log(marg_like+eps);
  % If there is no blocking, the normalization is simply due to the
  % likelihood computation:
end
log_lik_z = zeros(1,T);
Nz = zeros(Kz,Kz);
for tt = 1:T
    if tt > 1
        Nz(z(tt-1),z(tt)) = Nz(z(tt-1),z(tt)) + 1;
    end
    log_lik_z(tt) = log_block_like(z(tt),tt);
end
% Compute marginal for first time point
% fwd_msg = zeros(Kz,1);
log_pi_z = log(pi_z+eps);
log_pi_init = log(pi_init+eps);
log_oz_lik = sum(log_lik_z)+log_pi_init(z(1))+sum(sum(Nz.*log_pi_z));
oz_lik = exp(log_oz_lik-max(log_oz_lik));
% path = zeros(Kz,T);
% fwd_msg = log_pi_init+log_block_like(:,1)';
% for tt = 2:T
%     [fwd_msg,last_step] = max(repmat(fwd_msg',1,Kz)+log_pi_z,[],1);
%     fwd_msg = fwd_msg + log_block_like(:,tt)';
%     path(:,tt-1) = last_step';
% end
% optimal_path = zeros(1,T);
% [max_log_likelihood,optimal_path(T)] = max(fwd_msg);
% Nz = zeros(Kz,Kz);
% for tt = T-1:-1:1
%     optimal_path(tt) = path(optimal_path(tt+1),tt);
%     Nz(optimal_path(tt),optimal_path(tt+1)) = Nz(optimal_path(tt),optimal_path(tt+1)) + 1;
% end
% max_log_likelihood = max_log_likelihood - sum(sum(log_pi_z.*Nz))...
%     - log_pi_init(optimal_path(1)) + sum(loglike_normalizer);