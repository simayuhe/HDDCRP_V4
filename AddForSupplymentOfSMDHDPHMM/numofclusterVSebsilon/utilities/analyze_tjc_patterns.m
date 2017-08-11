function [lrn_acc1, lrn_acc2] = analyze_tjc_patterns(cluster, tjc_res, true_labels, do_plot, varargin)
if size(cluster,1)~=1
    cluster = cluster';
end
if size(true_labels,1)~=1
    true_labels = true_labels';
end
if do_plot
plot_type = 0;
if length(varargin) == 1
    axi_lim = varargin{1};
    plot_type = 1;
end
if length(varargin) == 2
    plot_type = 2;
    axi_lim = varargin{1};
    I = varargin{2};
end
unique_c = unique(cluster);
Kc = length(unique_c);
n_row = floor(sqrt(Kc));
n_col = ceil(Kc/n_row);
cnt = 0;
figure
for ii = unique_c
    cnt = cnt+1;
    subplot(n_row,n_col,cnt);
    switch plot_type
        case 0
            plot_tjcs(tjc_res,find(cluster == ii));
        case 1
            plot_tjcs(tjc_res,find(cluster == ii),axi_lim);
        case 2
            plot_tjcs(tjc_res,find(cluster == ii),axi_lim,I);
    end
    
    title(num2str(cnt));
end
end
[~, hamming_dist, ~, ~] = mapSequence2Truth(true_labels,cluster);
lrn_acc1 = 1 - hamming_dist;
lrn_acc2 = compute_lrn_acc(true_labels,cluster);
% fprintf(1, 'correct rate of dhdp-hmm: %f\n', 1-hamming_dist);
% [~, hamming_dist, ~, ~] = mapSequence2Truth(true_labels,init_c);
% fprintf(1, 'correct rate of sc-dtw: %f\n', 1-hamming_dist);
% fprintf(1, 'second correct rate of dhdp-hmm: %f\n', compute_lrn_acc(true_labels,cluster));
% fprintf(1, 'second correct rate of sc-dtw: %f\n', compute_lrn_acc(true_labels,init_c));

