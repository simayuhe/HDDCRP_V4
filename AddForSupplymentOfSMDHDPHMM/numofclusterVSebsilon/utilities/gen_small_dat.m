load('Syn_tjc.mat','tjc','labels');
oldC = max(labels);
newC = 5;
num_tjcs = length(tjc);
sel_labels = randperm(oldC);
sel_labels = sel_labels(1:newC);
sel_inds = cell(newC,1);
new_labels = cell(newC,1);
for ii = 1:newC
    sel_inds{ii} = find(labels==sel_labels(ii));
    new_labels{ii} = ones(length(sel_inds{ii}),1)*ii;
end
sel_inds = cell2mat(sel_inds);
new_labels = cell2mat(new_labels);
tjc = tjc(sel_inds);
labels = new_labels';
save('Syn_tjc_small.mat','tjc','labels');