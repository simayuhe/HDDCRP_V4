%������������ĵ�
clear all;close all;clc;
imsize = 5;%ͼƬ�ߴ�
noiselevel = 0.01;%��������
numbarpermix = [0 ones(1,3)]/3;%ʲô��;
% numbarpermix = [0 ones(1,2)]/2;
numcluster = 2;
numgroup     = 4;%�ĵ�����
numdata      = 20;%ÿƪ�ĵ��е�����Ŀ
% load trainss
[trainss,thetas] = genbars2(imsize,noiselevel,numbarpermix,numgroup,numdata,numcluster);