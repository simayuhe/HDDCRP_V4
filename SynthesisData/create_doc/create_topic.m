%%%��������,����Ĭ�ϲ���10�����⣬ϣ�����ӻ�֮����5*5�ķ����ÿ���к�ÿ����
%��ν��������һƪ�ĵ��и������ʳ��ֵĸ��ʴ�С�γɵ�һ����ɢ�ֲ������ｫ�ʵ䳤��Ĭ��Ϊ5*5���ûҶ�ͼ��ʾ����ÿ�����ʳ��ֵĸ���
close all;clear all;clc;
imsize=5;%���ӻ�֮���ͼƬ�ߴ�
num_topics=10;%������
num_words=25;%�ֵ䳤��
topic_matrix=zeros(num_words,num_topics);%�����洢����
noiselevel=0;%����������

%figure;
%colormap(gray);
for ii=1:imsize
   im = ones(imsize,imsize)*noiselevel/imsize^2;
  im(:,ii) = im(:,ii) + 1/imsize;
  topic_matrix(:,ii) = im(:)/sum(im(:));
end
for ii = 1:imsize
  im = ones(imsize,imsize)*noiselevel/imsize^2;
  im(ii,:) = im(ii,:) + 1/imsize;
  topic_matrix(:,imsize+ii) = im(:)/sum(im(:));
end
topic_matrix=topic_matrix*25;
fid=fopen('topics.txt','wt');
[row,col]=size(topic_matrix);
for i=1:1:col
    for j=1:1:row
        if j==row
            fprintf(fid ,'%g\n',topic_matrix(j,i));
        else
            fprintf(fid,'%g\t',topic_matrix(j,i));
        end
    end
end
fclose(fid);
file_name = sprintf('%s%s%d%s','topics.txt');
[topics, numclass] = readTopic(file_name, num_words);
imlayout(topics, [5 5 1 numclass],[min(topics(:)) max(topics(:))],'y');
