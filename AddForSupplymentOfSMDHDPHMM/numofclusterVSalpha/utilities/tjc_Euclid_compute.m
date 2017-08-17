function dist = tjc_Euclid_compute(tjc1, tjc2)
len1 = size(tjc1,1);
len2 = size(tjc2,1);
% we need len1 > len2
if len1 < len2
    tjc0 = tjc1;
    tjc1 = tjc2;
    tjc2 = tjc0;
    len0 = len1;
    len1 = len2;
    len2 = len0;
end
num_fold = floor(len1/len2);
num_remainder = rem(len1,len2);
num_interval = round(len2 / num_remainder); 
dist = 0;
num_adders = 0;
i1 = 0;
tjc2_remained = 0;
for i2 = 1:len2
    if i1 < len1
        for jj = 1:num_fold
            if i1 < len1
                i1 = i1 + 1;
                diff = tjc1(i1,:)-tjc2(i2,:);
                dist = dist + diff*diff';
                num_adders = num_adders + 1;
            else
                break;
            end
        end
    else
        tjc2_remained = 1;
        break;
    end
    if rem(i2,num_interval) == 0
        if i1 < len1
            i1 = i1 + 1;
            diff = tjc1(i1,:)-tjc2(i2,:);
            dist = dist + diff*diff';
            num_adders = num_adders + 1;
        end
    end
end
if i1 < len1;
    for i1 = i1+1:len1
        diff = tjc1(i1,:)-tjc2(end,:);
        dist = dist + diff*diff';
        num_adders = num_adders + 1;
    end
elseif tjc2_remained
    for i2 = i2:len2
        diff = tjc2(i2,:)-tjc1(end,:);
        dist = dist + diff*diff';
        num_adders = num_adders + 1;
    end
end
dist = dist / num_adders;