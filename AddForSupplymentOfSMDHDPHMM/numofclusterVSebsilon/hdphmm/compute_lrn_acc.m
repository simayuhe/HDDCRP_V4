function lrn_acc = compute_lrn_acc(true_labels, est_labels)
% compute learning accuracy
unique_est_labels = unique(est_labels);
K = length(unique_est_labels);
lrn_acc = 0;
for kk = unique_est_labels
    inds_logic = (est_labels == kk);
    Bk = sum(inds_logic);
    true_labels_temp = true_labels(inds_logic);
    unique_true_labels = unique(true_labels_temp);
    bk = sum(unique_true_labels(1)==true_labels_temp);
    for kkk = 2:length(unique_true_labels)
        bk_cand = sum(unique_true_labels(kkk)==true_labels_temp);
        if bk_cand > bk
            bk = bk_cand;
        end
    end
    lrn_acc = lrn_acc + bk/Bk;
end
lrn_acc = lrn_acc/K;    
