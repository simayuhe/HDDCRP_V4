%构造合成数据集并送入hddcrp v2.0中进行训练，训练之后在这里显示训练结果
%%
close all;clear all;clc;
load ('./get_documents/docs76.mat')
trainss=docs;
for i=1:1:length(trainss);
    p{i}=1:i;
    q{i}=zeros(1,i);
end
cand_links=p';
log_priors=q';

save ('D:\code\mycode\HDDCRP_V3\synthetic\inputdata\synthetic_dat_76.mat','cand_links','log_priors','trainss')