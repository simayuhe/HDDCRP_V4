function y = tjc_preprocess(x,ds,del)
ds2 = ds*ds;
ns = 3;
d0 = 1;
% ds = 3;
sd0 = d0*d0;
width = 5;
delta_angle = .5*pi;
m_max = 10000;
% delete points getting too close
if del
    del_set = zeros(1,size(x,1));
    % last_p = 1;
    thick = 0;
    start_p = 1;
    for ii = 2:size(x,1)
        diff = x(ii,:) - x(ii-1,:);
        if diff*diff' < sd0
            if thick == 0
                thick = 1;
                start_p = ii-1;
                del_set(ii) = 1;
            else
                diff1 = (x(start_p,:)-x(ii,:));
                if diff1*diff1' < 25
                    del_set(ii) = 1;
                else
                    thick = 0;
                end
            end
        end
    end
    x(del_set==1,:) = [];
    % y = x;
end
% add point to head and tail
nht = 5;
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
while delta_y1*delta_y1+delta_x1*delta_x1 < ds2 || abs(atan2(delta_y1,delta_x1)-ang0) > delta_angle    
    idx_next = idx_next + 1;
    cnt1 = cnt1 + 1;
    if idx_next <= size(x1,1)-hns && cnt1 < width
        delta_y1 = x1(idx_next,2) - y0(2);
        delta_x1 = x1(idx_next,1) - y0(1);
    else
        break;
    end
end
idx_last = idx_next + hns;

while(idx_last < size(x1,1))
    temp = [y(cnt-hns+1:cnt,:); x1(idx_next:idx_last,:)];
    [delta_y, delta_x, h, s] = ls_fit(temp);
    cnt = cnt + 1;
    y(cnt,:) = y0 + s * [delta_x delta_y] / h * ds;
    y0 = y(cnt,:);
    
    delta_y1 = x1(idx_next,2) - y0(2);
    delta_x1 = x1(idx_next,1) - y0(1);
    cnt1 = 0;
    ang0 = atan2(s*delta_y,s*delta_x);
    while delta_y1*delta_y1+delta_x1*delta_x1 < ds2 || abs(atan2(delta_y1,delta_x1)-ang0) > delta_angle
        idx_next = idx_next + 1;
        cnt1 = cnt1 + 1;
        if idx_next <= size(x1,1)-hns && cnt1 < width
            delta_y1 = x1(idx_next,2) - y0(2);
            delta_x1 = x1(idx_next,1) - y0(1);
        else
            break;
        end
    end
    idx_last = idx_next + hns;
end
y(cnt+1:end,:) = [];
y(1:hns,:) = [];

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