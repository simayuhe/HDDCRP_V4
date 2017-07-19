function [  ] = generate_docs( topic_file,num_doc,n_words )
%GENERATE_DOCS �˴���ʾ�йش˺����ժҪ
%   �˴���ʾ��ϸ˵��
topics=load(topic_file);
[num_topics,num_words]=size(topics);%�����num_wordsָ�����ֵ䳤��
docs=cell(num_doc,1);
for i=1:1:num_doc
    %����ÿƪ�ĵ��������õ����⼰Ȩ������ͬ��
    doc_name=['../data/doc_' num2str(i) '.txt'];%�������ÿƪ�ĵ�����һ�εķ�ʽ
    fid=fopen(doc_name,'wt');
    
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
        fprintf(fid,'%g\t',words);%д���ļ�
        clear words;
    end
    
    fclose(fid);%�ر��ļ�
end
save ('docs76.mat','docs');

end

