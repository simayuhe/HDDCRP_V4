%¥ ∆µø… ”ªØ
filename = 'doc_t.txt';
dict_size = 25;
freq=countwordfrequency(filename,dict_size);
%imlayout(topics, [5 5 1 numclass],[min(topics(:)) max(topics(:))],'y');
imlayout(freq,[sqrt(dict_size) sqrt(dict_size) 1 1],[min(freq(:)) max(freq(:))],'y');
