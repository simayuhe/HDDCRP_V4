function init_c = compute_init_c(D, Kc)

% sigma = 0;
% [init_c evd_time kmeans_time total_time] = sc(D, sigma, Kc);
% init_c = init_c';
 neighbor_num = 15;
 [u0,A_LS,u1] = scale_dist(D,floor(neighbor_num/2)); %% Locally scaled affinity matrix
 ZERO_DIAG = ~eye(size(D,1));
 A_LS = A_LS.*ZERO_DIAG;
 [clusts, ~] = gcut(A_LS,Kc);
 init_c = zeros(1, size(D,1));
 for jj = 1:length(clusts)
     init_c(clusts{jj}) = jj;
 end
