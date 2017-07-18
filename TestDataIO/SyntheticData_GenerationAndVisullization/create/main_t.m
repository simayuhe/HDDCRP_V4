%作为临时主函数，用来调试和调用各个正在编写的函数
close all;
clear all;
clc;
topic_file='topics.txt';
num_doc=36;
n_words=100;
generate_docs( topic_file,num_doc,n_words );
% filename = 'doc_2.txt';
% dict_size = 25;
% 
% freq=countwordfrequency(filename,dict_size);
% visualize_freq(freq,dict_size);
