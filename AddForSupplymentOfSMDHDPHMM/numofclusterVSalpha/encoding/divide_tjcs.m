function inds_sub_tjcs = divide_tjcs(save_dir,tjc_pro,len_tjc,do_plot,ave_num_seg,varargin)
if ~exist(save_dir,'file')
	mkdir(save_dir);
end
if save_dir(end)~='\' && save_dir(end)~='/'
    save_dir = [save_dir '/'];
end
% addpath('../ZPclustering/');
% addpath('../spectralclustering-1.1');

num_tjc = length(tjc_pro);
if ~isempty(varargin)
    ave_len_seg = varargin{1};
else
    ave_len_seg = sum(len_tjc) / (ave_num_seg*num_tjc);
end
inds_sub_tjcs = cell(length(tjc_pro),1);
if do_plot
    h1 = figure;
end
for ii = 1:length(tjc_pro)
    ii
    k = ceil(len_tjc(ii)/ave_len_seg);
    X = tjc_pro{ii};
    if k > 1
        X = [X 50*(1:size(X,1))'];
        D = sqrt(dist2(X,X));              %% Euclidean distance
%        [seg_labels, evd_time, kmeans_time, total_time] = sc(D, 0, k);

%             delta = sum(sum(sqrt(D)))/numel(D);
%         A_LS = exp(-D./(.02*delta*delta));
             neighbor_num = 15;
             [u0,A_LS,u1] = scale_dist(D,floor(neighbor_num/2)); %% Locally scaled affinity matrix
             ZERO_DIAG = ~eye(size(X,1));
             A_LS = A_LS.*ZERO_DIAG;
             [clusts, ~] = gcut(A_LS,k);
			 seg_labels = zeros(1, size(X,1));
			 for jj = 1:length(clusts)
				 seg_labels(clusts{jj}) = jj;
			 end

        gaps = zeros(1,100);
        cnt = 1;
        gaps(1) = 1;
        idx_cur = seg_labels(1);
        for jj = 2:length(seg_labels)
            if seg_labels(jj) ~= idx_cur
                idx_cur = seg_labels(jj);
                cnt = cnt + 1;
                gaps(cnt) = jj;
            end
        end
        k = cnt;
        gaps(cnt+1) = length(seg_labels);
        inds_sub_tjcs{ii} = cell(1,k);
        inds_sub_tjcs{ii}{1} = gaps(1):gaps(2);
        for kk = 2:k
            inds_sub_tjcs{ii}{kk} = gaps(kk)-1:gaps(kk+1);
        end

%         CLUSTER_NUM_CHOICES = k-2:k+2;
%         [clusts_RLS, rlsBestGroupIndex, qualityRLS] = cluster_rotate4dividing(A_LS,CLUSTER_NUM_CHOICES,0,1);
%         clusts = clusts_RLS{rlsBestGroupIndex};
%         k = length(clusts);

% sort the clusters!!!!!!!!!!!!!!!!!!!!!
%             idx_min = zeros(1,k);
%             for jj = 1:k
%                 idx_min(jj) = clusts{jj}(1);
%             end
%             [~, idx_sort] = sort(idx_min);
%             clusts = clusts(idx_sort);
% 
%             inds_sub_tjcs{ii} = clusts;
    else
        inds_sub_tjcs{ii} = cell(1,1);
        inds_sub_tjcs{ii}{1} = 1:size(X,1);
    end
    if do_plot
        plot_divided_tjc(tjc_pro{ii},inds_sub_tjcs{ii},h1);
        input('Press enter to continue:');
        figure(h1);
    end
end
save([save_dir 'inds_sub_tjcs.mat'], 'inds_sub_tjcs');	
