function plot_tjcs(tjc,idx,varargin)
if size(idx,1)~=1
    idx = idx';
end
if length(varargin)>0
	axis(varargin{1});
end
if length(varargin)>1
	I = varargin{2};
	I = flipdim(I,1);
	imshow(I);
	axis xy
end
hold on
for ii = idx
    tjc_ii = tjc{ii};
    plot(tjc_ii(1:end-5,1),tjc_ii(1:end-5,2));
    plot(tjc_ii(end-5:end,1),tjc_ii(end-5:end,2),'r');
end
