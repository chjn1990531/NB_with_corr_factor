function [y] = pos_encoding(label,dif_label)
%POS_ENCODING input label to get encoding y, label should be column vector
% 
n = size(label,1);  % number of samples
k = size(dif_label,1);  % number classes
y = zeros(n,k);
for i = 1:n
    y(i,find(dif_label==label(i))) = 1;
end

