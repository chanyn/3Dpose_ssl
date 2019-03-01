function vis2Dskel(W,skel,linestyle,linewidth)

if nargin < 3
    linestyle = '-';
end

if nargin < 4
    linewidth = 1.5;
end

connect = skelConnectionMatrix(skel);
indices = find(connect);
[I, J] = ind2sub(size(connect), indices);

% markersize = 5;
% plot(W(1,:),W(2,:),'o','MarkerSize',markersize,...
%     'MarkerEdgeColor',color,'MarkerFaceColor',color);
% hold on
% for i =1:size(W,2)
%     text(W(1,i),W(2,i),int2str(i));
% end
for i = 1:length(indices)
    line([W(1,I(i)) W(1,J(i))], ...
         [W(2,I(i)) W(2,J(i))],'color',skel.tree(I(i)).color,'LineWidth',linewidth,'LineStyle',linestyle);
end

axis equal off
% ylim([1.2*min(W(2,:)) 1.2*max(W(2,:))]);

end


function connection = skelConnectionMatrix(skel)

connection = zeros(length(skel.tree));
for i = 1:length(skel.tree);
    for j = 1:length(skel.tree(i).children)
        connection(i, skel.tree(i).children(j)) = 1;
    end
end

end
