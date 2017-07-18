%由现有主题产生num_doc篇文档，每篇文档n_words
close all;clear all;clc
load topics.txt
[num_topics,num_words]=size(topics);
num_doc=76;%
n_words=110;
docs=cell(num_doc,1);
for i=1:1:num_doc
    %对于每篇文档其混合所用的主题及权重是相同的
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
        clear words;
    end
end
save ('docs76.mat','docs');
%可视化
classqq=zeros(num_doc,num_words);%统计单词个数,文档个数*字典长度
fid=fopen('docs76.txt','wt');
for i=1:1:length(docs)
    for j=1:1:length(docs{i,1})
        word=docs{i,1}(1,j);
        classqq(i,word)=classqq(i,word)+1;
        if j==length(docs{i,1})
            fprintf(fid ,'%g\n',word);
        else
            fprintf(fid,'%g\t',word);
        end
        clear word;
    end
end
fclose(fid);
fid=fopen('classqq_docs.txt','wt');
for i=1:1:num_doc
    for j=1:1:num_words
          if j==num_words
            fprintf(fid ,'%g\n',classqq(i,j));
        else
            fprintf(fid,'%g\t',classqq(i,j));
        end
    end
end
fclose(fid);
file_name = sprintf('%s%s%d%s','classqq_docs.txt');
[topics, numclass] = readTopic(file_name, num_words);
imlayout(topics, [5 5 1 numclass],[min(topics(:)) max(topics(:))],'y');