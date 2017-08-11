close all;clear all;
load('D:\code\mycode\supplyment_smdhdphmm\numofclusterVSebsilon\results\CASIA\code.mat')
holdout_index=randperm(1500,50);
holdout_data=codes(holdout_index);
for i=1:1:length(holdout_data)
    holdout_data{i,1}=round(holdout_data{i,1}+randn(size(holdout_data{i,1},1),size(holdout_data{i,1},2)));
    holdout_data{i,1}(holdout_data{i,1}>200)=200;
    holdout_data{i,1}(holdout_data{i,1}<1)=1;
end
save('holdout_data.mat','holdout_data');