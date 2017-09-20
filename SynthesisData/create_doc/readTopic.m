function [topics, nTopic] = readTopic(fileName,nWord)
fid = fopen(fileName, 'r');
% nWord=4*nCell(1)*nCell(2)/100;

W = fscanf(fid,'%d');
nTopic=length(W)/nWord;
topics = reshape(W, nWord, nTopic);
% topics=zeros(nTopic,nWord);
% for i=1:nTopic
%     topics(i,:)=W((i-1)*nWord+1:i*nWord);
% 
% end
fclose(fid);
end