function y = tjc_preprocess2(x,n)
% resample an equal interval trajectory into n points
len = 0;
for ii = 2:size(x,1)
    len = len + norm(x(ii,:)-x(ii-1,:));
end
if len == 0
    y = repmat(x,n,1);
    return;
end
ds = len / n;
ns = 3;
m_max = 10000;
% delete points getting too close
% add point to head and tail
nht = 5;
if size(x,1)<nht
    nht = size(x,1);
end
head = x(nht:-1:1,:);
tail = x(end-nht+1:end,:);
hns = floor(ns/2);
x1 = zeros(size(x,1)+hns*2,2);
x1(hns+1:end-hns,:) = x;
[delta_y, delta_x, h, s] = ls_fit(tail);
for ii = 1:hns
    x1(end-hns+ii,:) = x(end,:) + ii * s * [delta_x delta_y] / h * ds;
end
[delta_y, delta_x, h, s] = ls_fit(head);
for ii = 1:hns
    x1(ii,:) = x(1,:) + (hns-ii+1) * s * [delta_x delta_y] / h * ds;
end
s = -s;
% interpolate
y = zeros(m_max,2);
y(1:hns+1,:) = x1(1:hns+1,:);
cnt = hns+1;
y0 = y(hns+1,:);
idx_next = hns+2;

delta_y1 = x1(idx_next,2) - y0(2);
delta_x1 = x1(idx_next,1) - y0(1);
cnt1 = 0;
ang0 = atan2(s*delta_y,s*delta_x);
while delta_y1*delta_y1+delta_x1*delta_x1 < ds || abs(atan2(delta_y1,delta_x1)-ang0) > .5*pi
    idx_next = idx_next + 1;
    cnt1 = cnt1 + 1;
    if idx_next <= size(x1,1)-hns && cnt1 < 5
        delta_y1 = x1(idx_next,2) - y0(2);
        delta_x1 = x1(idx_next,1) - y0(1);
    else
        break;
    end
end
idx_last = idx_next + hns;

while(cnt < n+hns)
    if idx_next > size(x1,1)
        idx_next = size(x1,1);
    end
    if idx_last > size(x1,1)
        idx_last = size(x1,1);
    end        
    temp = [y(cnt-hns+1:cnt,:); x1(idx_next:idx_last,:)];
    [delta_y, delta_x, h, s] = ls_fit(temp);
    cnt = cnt + 1;
    y(cnt,:) = y0 + s * [delta_x delta_y] / h * ds;
    y0 = y(cnt,:);
    
    delta_y1 = x1(idx_next,2) - y0(2);
    delta_x1 = x1(idx_next,1) - y0(1);
    cnt1 = 0;
    ang0 = atan2(s*delta_y,s*delta_x);
    while delta_y1*delta_y1+delta_x1*delta_x1 < ds || abs(atan2(delta_y1,delta_x1)-ang0) > .5*pi
        idx_next = idx_next + 1;
        cnt1 = cnt1 + 1;
        if idx_next <= size(x1,1)-hns && cnt1 < 5
            delta_y1 = x1(idx_next,2) - y0(2);
            delta_x1 = x1(idx_next,1) - y0(1);
        else
            break;
        end
    end
    idx_last = idx_next + hns;
end
y(1:hns,:) = [];
y(n+1:end,:) = [];

function [delta_y, delta_x, h, s] = ls_fit(x)
x_bar = mean(x(:,1));
delta_y = sum((x(:,1)-x_bar).*x(:,2));
delta_x = (x(:,1)-x_bar)'*(x(:,1)-x_bar);
h = sqrt(delta_y*delta_y + delta_x*delta_x);
n = size(x,1);
hn = floor(n/2);
dx = sum(x(end-hn+1:end,1)) - sum(x(1:hn,1));
dy = sum(x(end-hn+1:end,2)) - sum(x(1:hn,2));
if abs(dx) > abs(dy)
    s = sign(dx);
else
    s = sign(dy) * sign(delta_y);
end