function  visualize_freq( freq,dict_size )
%VISUALIZE_FREQ 此处显示有关此函数的摘要
%   此处显示详细说明
%对imlayout的一个简单包装，每次只显示一个词频的频谱
imlayout(freq,[sqrt(dict_size) sqrt(dict_size) 1 1],[min(freq(:)) max(freq(:))],'y');
end

