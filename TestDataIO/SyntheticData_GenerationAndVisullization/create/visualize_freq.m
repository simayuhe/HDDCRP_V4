function  visualize_freq( freq,dict_size )
%VISUALIZE_FREQ �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
%��imlayout��һ���򵥰�װ��ÿ��ֻ��ʾһ����Ƶ��Ƶ��
imlayout(freq,[sqrt(dict_size) sqrt(dict_size) 1 1],[min(freq(:)) max(freq(:))],'y');
end

