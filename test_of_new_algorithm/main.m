%% data loading 
clear
clc
close all

% % reuter's data
% load('feature')
% load('label')
% 
% 20 news data
load('20_feature');
load('20_label');

feature = feature + 10^-8;

train_size = 0.1;   % percentage of data for training

feature = feature + 10^-8;

dif_label = unique(label);
[n_sample, ~] = size(feature);
train_ind = randperm(n_sample, round(train_size*n_sample));
test_ind = setdiff(1:n_sample, train_ind);

train_feature = feature(train_ind,:);
train_label = label(train_ind);
test_feature = feature(test_ind,:);
test_label = label(test_ind);


% add in negative features
% neg_size = 0.9;
% neg_ind = randperm(n_sample, round(neg_size*n_sample));
% neg_feature = feature(neg_ind,:);
% neg_label = label(neg_ind,:);


clear n_sample feature label neg_ind pos_ind pos_size train_size
clear train_ind test_ind


%% encoding
n_label = size(dif_label,1);
for i = 1:n_label
    % first encoding
    y = pos_encoding(train_label,dif_label);
    z = 1-y;
%     z = neg_encoding(neg_label,dif_label);
end


%% compare our algorithm with nb

theta_ours_L2 = our_method_L2(train_feature,y,train_feature,z,dif_label);

% theta_ours_L1 = our_method_L1(train_feature,y,neg_feature,z,dif_label);
% theta_ours_L2 = our_method_L2(train_feature,y,neg_feature,z,dif_label);

theta_all = zeros(size(dif_label,1),size(train_feature,2));
for i = 1:size(dif_label,1)
    i
    ind = find(train_label==dif_label(i));
    theta_all(i,:) = sum(train_feature(ind,:));
end
theta_all = theta_all./sum(theta_all,2);

%% get accuracy
% clear all
% clc
% load thetas.mat;

% accu_all = get_accuracy(theta_all,test_feature,test_label);
% accu_ours_L1 = get_accuracy(theta_ours_L1,test_feature,test_label);
% accu_ours_L2 = get_accuracy(theta_ours_L2,test_feature,test_label);

% training set behavior

accu_all = get_accuracy(theta_all,train_feature,train_label);
accu_ours_L2 = get_accuracy(theta_ours_L2,train_feature,train_label);
%%
close all
figure();
plot(accu_all,'*');
% hold on
% plot(accu_ours_L1,'o');
hold on
plot(accu_ours_L2,'x');
mean_all = refline([0, mean(accu_all)]);
% mean_ours_L1 = refline([0, mean(accu_ours_L1)]);
mean_ours_L2 = refline([0, mean(accu_ours_L2)]);
mean_all.Color = 'g';
% mean_ours_L1.Color = 'm';
mean_ours_L2.Color = 'c';

% l = legend('$\hat{\theta}$','$\hat{\theta}^{L_1}$','$\hat{\theta}^{L_2}$',...
%     'average $\hat{\theta}$','average $\hat{\theta}^{L_1}$', ...
%     'average $\hat{\theta}^{L_2}$');

l = legend('$\hat{\theta}$','$\hat{\theta}^{L_2}$',...
    'average $\hat{\theta}$', 'average $\hat{\theta}^{L_2}$');

% l = legend('$\hat{\theta}^{L_1}$','$\hat{\theta}^{L_2}$',...
%     'average $\hat{\theta}^{L_1}$', 'average $\hat{\theta}^{L_2}$');

set(l,'Interpreter','Latex');
% plot(accu_ours,'o');






