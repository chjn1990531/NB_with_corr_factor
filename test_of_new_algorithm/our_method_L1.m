function [theta] = our_method_L1(pos_feature,y,neg_feature,z,dif_label)
%OUR_METHOD return centroid of each class, stored in theta
% 

k = size(dif_label,1);
v = size(pos_feature,2);
theta = zeros(k,v);
for i = 1:k
    i
    ind_pos = find(y(:,i)==1);
    Z = (1-z)./(k-sum(z,2));
    ind_neg = find(Z(:,i)~=0);
    tmp = Z(ind_neg,i);
    if isempty(tmp)==0
        tmp = tmp(1);
    else
        tmp = 0;
    end
    
    theta(i,:) = sum(pos_feature(ind_pos,:))+sum(tmp*neg_feature(ind_neg,:));
end
% ind = find(theta<0);
% theta(ind) = 10^-8;
theta = theta./sum(theta,2);
end


