function y = tjc_denoise4(x,n)
% median filter
hn = floor(n/2);
y = x;
for ii = hn+1:size(x,1)-hn
    y(ii,:) = median(x(ii-hn:ii+hn,:));
end