function [theta] = our_method_L2(pos_feature,y,neg_feature,z,dif_label)
%OUR_METHOD return centroid of each class, stored in theta
% 

k = size(dif_label,1);
v = size(pos_feature,2);
theta = zeros(k,v);
for i = 1:k
    i
    ind_pos = find(y(:,i)==1);
    ind_neg = find(z(:,i)==1);
    theta(i,:) = 2*(sum(pos_feature)+sum(neg_feature))+...
    sum(pos_feature(ind_pos,:))-sum(neg_feature(ind_neg,:));
end
% ind = find(theta<0);
% theta(ind) = 10^-8;
theta = theta./sum(theta,2);
end


