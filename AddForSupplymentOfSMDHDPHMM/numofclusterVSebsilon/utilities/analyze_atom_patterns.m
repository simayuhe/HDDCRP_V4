% load('tjc_pro.mat');
% clear;
close all;
load('./results/CASIA/res_sub_tjc.mat');
% load('../HMMdata4ASL/exp1/HDPHMMDPstatsiter400trial1.mat');

X = res_sub_tjc;
X = X.';
X = reshape(X,size(X,1)/2,size(X,2)*2);
X = mat2cell(X,size(X,1),2*ones(1,size(X,2)/2));
X = X.';
stateSeq = S.stateSeq;
idx = 2;
Ns = S.stateCounts(idx).Ns;
Ns = sum(Ns,2);
[~,inds] = sort(Ns,'descend');
labels = zeros(length(X),1);
% load('init_z.mat','init_z','Kz');
cnt = 0;
for ii = 1:size(stateSeq,2)
%     z = init_z{ii};
    z = stateSeq(idx,ii).z;
    labels(cnt+1:cnt+length(z)) = z;
    cnt = cnt+length(z);
end
unique_z = unique(labels);
Kz = length(unique_z);
if Kz > 49
    Kz = 49;
    unique_z = inds(1:49);
end
% Kz = 20;
n_row = floor(sqrt(Kz));
n_col = ceil(Kz/n_row);
cnt = 0;
for ii = unique_z'
    cnt = cnt+1;
    subplot(n_row,n_col,cnt);
    plot_tjcs(X,find(labels==ii)');
    title(num2str(cnt));
end