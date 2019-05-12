% reuters data

clear all
clc

load('correct.mat');
load('original_mat.mat');

label_ind = [];
for i = 1:66
    if size(find(correct==i),1)>100
        label_ind = [label_ind,i];
    end
end
% label_ind = [7,8,9,11,14,17,18];



ind = [];
for i = 1:size(label_ind,2)
    ind = union(ind,find(correct==label_ind(i)));
end

feature = M(ind,:);
label = correct(ind,:);

save feature.mat feature;
save label.mat label;