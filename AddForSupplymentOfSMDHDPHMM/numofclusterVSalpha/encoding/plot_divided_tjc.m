function plot_divided_tjc(x,clusts,fig)
figure(fig);
hold off;
clf
hold on
for jj = 1:2:length(clusts)
    x_jj = x(clusts{jj},:);
    plot(x_jj(:,1),x_jj(:,2),'.-','LineWidth',2,'Color',[0 0 1]);
%     draw_arrow(x_jj,fig);
end
for jj = 2:2:length(clusts)
    x_jj = x(clusts{jj},:);
    plot(x_jj(:,1),x_jj(:,2),'.-','LineWidth',2,'Color',[1 0 0]);
%     draw_arrow(x_jj,fig);
end