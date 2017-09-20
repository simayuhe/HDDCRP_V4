function plot_panorama(tjcs, labels, varargin)
if ~iscell(labels)
    if size(labels,1)~=1
        labels = labels';
    end
    C = max(labels);
else
    C = length(labels);
end
M = floor(sqrt(C));
N = ceil(sqrt(C));
for cc = 1:C
    subplot(M,N,cc);
    if iscell(labels)
        plot_tjcs(tjcs,labels{cc},varargin{1,:});
    else
        plot_tjcs(tjcs,find(labels==cc),varargin{1,:});
    end
end