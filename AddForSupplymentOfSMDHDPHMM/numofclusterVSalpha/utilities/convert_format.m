load('SynTraj.mat','trajStruct');
tjc = struct2cell(trajStruct);
labels= tjc(2,:)';
labels = cell2mat(labels);
tjc = tjc(1,:)';
save('Syn_tjc.mat','tjc','labels');