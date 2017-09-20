close all;
clear all;
clc;

num_doc=360;

for i=1:1:num_doc;
    p{i}=1:i;
    q{i}=zeros(1,i);
end
cand_links=p';
log_priors=q';
for i=1:1:num_doc
cand_link_name=['../data/cand_link' num2str(i) '.txt'];
log_prior_name=['../data/log_prior' num2str(i) '.txt'];
fid1=fopen(cand_link_name,'wt');
fid2=fopen(log_prior_name,'wt');
 fprintf(fid1,'[%g]\t',length(p{i}));
 fprintf(fid2,'[%g]\t',length(q{i}));
for j=1:1:length(p{i})
 fprintf(fid1,'%g\t',p{i}(j));
 fprintf(fid2,'%g\t',q{i}(j));
end
fclose(fid1);
fclose(fid2);
end