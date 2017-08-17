function [neglog_c likelihood log_normalizer]= observation_likelihood(data_struct,obsModelType,dist_struct,theta)

pi_init = dist_struct.pi_init;
pi_z = dist_struct.pi_z;
pi_s = dist_struct.pi_s;
Kc = size(pi_z,3);
Kz = size(pi_z,1);
Ks = size(pi_s,2);

blockEnd = data_struct.blockEnd;
T = length(blockEnd);

% Compute likelihood matrix:
[likelihood log_normalizer] = compute_likelihood(data_struct,theta,obsModelType,Kz,Ks);

neglog_c = zeros(1,Kc);
for cc = 1:Kc
    % Pass messages forward to integrate over the mode/state sequence:
    [~, neglog_c_t] = forward_message_vec(likelihood,log_normalizer,blockEnd,pi_z(:,:,cc),pi_s,pi_init(:,cc)');
    neglog_c(cc) = sum(neglog_c_t);
end

return;
