function display_topics(res_dir, iter)
if(res_dir(end)~='/' || res_dir(end)~='\')
    res_dir=[res_dir '/'];
end
n_words = 25;
file_name = sprintf('%s%s%d%s',res_dir,'classqq',iter,'.txt');
[topics, numclass] = readTopic(file_name, n_words);
imlayout(topics, [5 5 1 numclass],[min(topics(:)) max(topics(:))],'y');