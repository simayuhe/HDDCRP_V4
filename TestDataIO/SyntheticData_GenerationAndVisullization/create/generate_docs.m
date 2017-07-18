function [  ] = generate_docs( topic_file,num_doc,n_words )
%GENERATE_DOCS 此处显示有关此函数的摘要
%   此处显示详细说明
topics=load(topic_file);
[num_topics,num_words]=size(topics);%这里的num_words指的是字典长度
docs=cell(num_doc,1);
for i=1:1:num_doc
    %对于每篇文档其混合所用的主题及权重是相同的
    doc_name=['..\data\doc_' num2str(i) '.txt'];%这里采用每篇文档保存一次的方式
    fid=fopen(doc_name,'wt');
    
    num_TopicSelected=randi(num_topics);%主题数目
    TopicSelected=randperm(num_topics,num_TopicSelected);
    temp=rand(1,num_TopicSelected);
    Weight=temp/sum(temp);
    clear temp;
    TopicDistribution=Weight*topics(TopicSelected,:);
    TopicDistribution=TopicDistribution/sum(TopicDistribution);%归一化得到新的离散分布
    for j=1:1:n_words
        %离散分布中采样n_words个单词组成文档
        words=randmult(TopicDistribution);
        docs{i,1}=[docs{i,1} words];
        fprintf(fid,'%g\t',words);%写入文件
        clear words;
    end
    
    fclose(fid);%关闭文件
end
save ('docs76.mat','docs');

end

