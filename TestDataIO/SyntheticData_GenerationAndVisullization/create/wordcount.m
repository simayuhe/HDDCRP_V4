%����һƪ�ĵ���ͳ�Ƹ������ʵĸ���
load doc_t.txt
dictionary_size = max(doc_t);%����û����0�Ĵ���
frequency = zeros(1,dictionary_size);
for i=1:1:length(doc_t)
    frequency(doc_t(i))=frequency(doc_t(i))+1;
end
frequency=frequency./length(doc_t);