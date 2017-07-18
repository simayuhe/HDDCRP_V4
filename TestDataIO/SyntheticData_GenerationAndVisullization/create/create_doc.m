%�������������num_docƪ�ĵ���ÿƪ�ĵ�n_words
close all;clear all;clc
load topics.txt
[num_topics,num_words]=size(topics);
num_doc=76;%
n_words=110;
docs=cell(num_doc,1);
for i=1:1:num_doc
    %����ÿƪ�ĵ��������õ����⼰Ȩ������ͬ��
    num_TopicSelected=randi(num_topics);%������Ŀ
    TopicSelected=randperm(num_topics,num_TopicSelected);
    temp=rand(1,num_TopicSelected);
    Weight=temp/sum(temp);
    clear temp;
    TopicDistribution=Weight*topics(TopicSelected,:);
    TopicDistribution=TopicDistribution/sum(TopicDistribution);%��һ���õ��µ���ɢ�ֲ�
    for j=1:1:n_words
        %��ɢ�ֲ��в���n_words����������ĵ�
        words=randmult(TopicDistribution);
        docs{i,1}=[docs{i,1} words];
        clear words;
    end
end
save ('docs76.mat','docs');
%���ӻ�
classqq=zeros(num_doc,num_words);%ͳ�Ƶ��ʸ���,�ĵ�����*�ֵ䳤��
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