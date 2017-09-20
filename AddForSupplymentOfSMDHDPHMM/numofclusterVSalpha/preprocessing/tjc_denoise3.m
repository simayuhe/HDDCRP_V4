function y = tjc_denoise3(x,n)
% DFT filter
X = fft(x);
Y = zeros(size(X));
Y(1:n,:) = X(1:n,:);
Y(end-n+1:end,:) = Y(end-n+1:end,:);
y = real(ifft(Y));