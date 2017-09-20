function [HoG H] = compute_HoG(x,n)
% n should be n = 2^m
HoG = zeros(1,n);
dx = x(2:end,:) - x(1:end-1,:);
angles = atan2(dx(:,2),dx(:,1))';
base = -pi-pi/n;
interval = 2*pi/n;
for ii = 1:length(angles)
    bin = floor((angles(ii)-base)/interval)+1;
    if bin > n
        bin = 1;
    end
    HoG(bin) = HoG(bin) + 1;
end
H = HoG;
HoG = HoG / sum(HoG);