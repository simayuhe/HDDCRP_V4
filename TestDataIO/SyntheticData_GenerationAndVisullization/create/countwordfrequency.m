function [ frequency ] = countwordfrequency( doc_name,dictionary_size )
%COUNTWORDFREQUENCY �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
doc_t=load(doc_name);
frequency = zeros(1,dictionary_size);
for i=1:1:length(doc_t)
    frequency(doc_t(i))=frequency(doc_t(i))+1;
end
frequency=frequency./length(doc_t);
end

