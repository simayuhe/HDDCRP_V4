function dist_mat = compute_distance_matrix(path_results_root, tjcs, metric)
% compute distance matrix
addpath('.\DynamicTimeWarping\');
total_num_tjcs = length(tjcs);
dist_mat = zeros(total_num_tjcs,total_num_tjcs);
cnt = 0;
tic
for ii = 1:total_num_tjcs
    for jj = ii+1:total_num_tjcs
        cnt = cnt + 1;
        fprintf(1,'%d %f\n', cnt, toc);
        switch metric
            case 'Euclid'
                dist_mat(ii,jj) = tjc_Euclid_compute(tjcs{ii},tjcs{jj});
            case 'DTW'
                dist_mat(ii,jj) = tjc_DTW_compute(tjcs{ii},tjcs{jj});
            case 'DTW2'
                comp_rate = 1;
                dist_mat(ii,jj) = tjc_DTW_compute2(tjcs{ii},tjcs{jj},comp_rate);
        end
    end
end

for ii = 2:total_num_tjcs
    for jj = 1:ii-1
        dist_mat(ii,jj) = dist_mat(jj,ii);
    end
end

save([path_results_root metric '_dist_mat.mat'],'dist_mat');