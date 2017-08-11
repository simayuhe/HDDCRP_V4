function dist = tjc_DTW_compute2(tjc1, tjc2, comp_rate)
% compute distance matrix M
r = size(tjc1,1);
c = size(tjc2,1);
M = zeros(r,c);
for ii = 1:r
    for jj = 1:c
        diff = tjc1(ii,:)-tjc2(jj,:);
        M(ii,jj) = diff * diff';
    end
end

C = [1 1 1.0;0 1 1.0;1 0 1.0];
[D,phi] = dpcore(M,C);
% ii = r;
% jj = c;
% K = 1;
% while ii > 1 && jj > 1
%   tb = phi(ii,jj);
%   ii = ii - C(tb,1);
%   jj = jj - C(tb,2);
%   K = K+1;
% end
% [~,~,D,sc] = dpfast(M);
dist = sqrt(D(end,end)/comp_rate);