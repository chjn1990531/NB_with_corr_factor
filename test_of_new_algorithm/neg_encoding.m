function [z] = neg_encoding(label,dif_label)
%NEG_ENCODING input label output encoding z, negative is given by random
%   Detailed explanation goes here
n = size(label,1);  % number of samples
k = size(dif_label,1);  % number classes
z = zeros(n,k);
for i = 1:n
    temp_set = setdiff(dif_label,label(i));
    z(i,find(dif_label==temp_set(randperm(k-1,1)))) = 1;
end