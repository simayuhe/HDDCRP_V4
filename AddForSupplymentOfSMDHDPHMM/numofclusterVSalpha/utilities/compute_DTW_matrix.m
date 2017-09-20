function D = compute_DTW_matrix(tjc)
n = length(tjc);
n_pairs = n*(n-1)/2;
idx_mat = zeros(n_pairs,2);
cnt = 0;
for ii = 1:n
    for jj = ii+1:n
        cnt = cnt + 1;
        idx_mat(cnt,:) = [ii jj];
    end
end
%     load('idx_mat.mat','idx_mat');
D = zeros(n,n);
D_temp = zeros(n_pairs,1);
matlabpool open
%     tic;
parfor nn = 1:n_pairs
    ii = idx_mat(nn,1);
    jj = idx_mat(nn,2);
    D_temp(nn) = tjc_DTW_compute2(tjc{ii},tjc{jj},1);
%         nn
%         fprintf(1,'%d average time cost: %f\n', nn, toc/nn);
%         toc;
end
matlabpool close
for nn = 1:n_pairs
    ii = idx_mat(nn,1);
    jj = idx_mat(nn,2);
    D(ii,jj) = D_temp(nn);
    D(jj,ii) = D_temp(nn);
end