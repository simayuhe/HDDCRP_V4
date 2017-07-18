%给出一篇文档，统计各个单词的个数
load doc_t.txt
dictionary_size = max(doc_t);%这里没考虑0的存在
frequency = zeros(1,dictionary_size);
for i=1:1:length(doc_t)
    frequency(doc_t(i))=frequency(doc_t(i))+1;
end
frequency=frequency./length(doc_t);