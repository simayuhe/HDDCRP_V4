%%%产生主题,这里默认产生10个主题，希望可视化之后是5*5的方格的每个行和每个列
%所谓的主题是一篇文档中各个单词出现的概率大小形成的一个离散分布，这里将词典长度默认为5*5，用灰度图表示其中每个单词出现的概率
close all;clear all;clc;
imsize=5;%可视化之后的图片尺寸
num_topics=10;%主题数
num_words=25;%字典长度
topic_matrix=zeros(num_words,num_topics);%用来存储主题
noiselevel=0;%不加入噪声

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
