%根据主题产生文档
clear all;close all;clc;
imsize = 5;%图片尺寸
noiselevel = 0.01;%加入噪声
numbarpermix = [0 ones(1,3)]/3;%什么用途
% numbarpermix = [0 ones(1,2)]/2;
numcluster = 2;
numgroup     = 4;%文档个数
numdata      = 20;%每篇文档中单词数目
% load trainss
[trainss,thetas] = genbars2(imsize,noiselevel,numbarpermix,numgroup,numdata,numcluster);