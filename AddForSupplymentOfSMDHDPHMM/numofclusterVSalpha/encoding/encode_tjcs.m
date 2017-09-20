function codes = encode_tjcs(features,inds_sub_tjcs,K,do_sc,varargin)
if isempty(varargin)
	if do_sc
		gen_nn_distance(features, 15, 100, 0);
		load 15_NN_sym_distance.mat
		% A = sqrt(dist2(features,features));
		[inds evd_time kmeans_time total_time] = sc(A, 0, K);
	else
		inds = k_means(features,[],K);
%         inds = kmeans(features,K);
	end
else
    centers = varargin{1};
    inds = zeros(size(features,1),1);
    for ii = 1:size(features,1)
        diff = repmat(features(ii,:),K,1) - centers;
        [~,inds(ii)] = min(sum(diff.*diff,2));
    end
end
    
	codes = cell(length(inds_sub_tjcs),1);
	cnt = 0;
	for ii = 1:length(inds_sub_tjcs)
		num_sub_ii = length(inds_sub_tjcs{ii});
		codes{ii} = inds(cnt+1:cnt+num_sub_ii)';
		cnt = cnt+num_sub_ii;
	end
