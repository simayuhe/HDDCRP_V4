%test for randmult.m
close all;
clear all;
clc;
pp=[0 0 10 1 0 0 1 0 0 0];
a=zeros(1,10);
for k=1:1:100
    j=randmult(pp,1);
   % a(j)=a(j)+1;
end
plot([1:10],j);