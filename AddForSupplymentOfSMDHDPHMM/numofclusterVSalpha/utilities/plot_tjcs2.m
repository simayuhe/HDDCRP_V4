function plot_tjcs2(tjc,idx,color)
for ii = idx
    tjc_ii = tjc{ii};
    n_tail = 3;
    plot(tjc_ii(1:end-n_tail,1),tjc_ii(1:end-n_tail,2),'Color',color);
    plot(tjc_ii(end-n_tail:end,1),tjc_ii(end-n_tail:end,2),'r');
%     text(tjc_ii(end,1),tjc_ii(end,2),num2str(ii));
end