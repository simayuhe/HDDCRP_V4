%��Ƶ���ӻ�
filename ='../hddcrp_txt/result_360/classqq100.txt';% '../..\result\classqq2.txt';
dict_size = 25;
class_all=load(filename);
for i=1:1:size(class_all,1)
    h=figure(i);
    imlayout(class_all(i,:),[sqrt(dict_size) sqrt(dict_size) 1 1],[min(class_all(i,:)) max(class_all(i,:))],'y');
    set(h,'visible','off');
str=sprintf('../result_figure_1/figure(%d)',i);
saveas(h,str,'jpg');
end
% freq=countwordfrequency(filename,dict_size);
% %imlayout(topics, [5 5 1 numclass],[min(topics(:)) max(topics(:))],'y');
% imlayout(freq,[sqrt(dict_size) sqrt(dict_size) 1 1],[min(freq(:)) max(freq(:))],'y');
