%����ϳ����ݼ�������hddcrp v2.0�н���ѵ����ѵ��֮����������ʾѵ�����
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