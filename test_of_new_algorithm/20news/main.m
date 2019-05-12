clear;
clc;
close all;

%% load data
train_size = 0.8;
[train_DT, test_DT, train_labels, test_labels] = load_data(train_size);

unique_labels = union(unique(train_labels), unique(test_labels)).';

%% one-hot encoding for positive labels and negative labels
% positive label: the actual label of a document
% negative label: a label that a document can not take value on

tmp = zeros(size(train_labels));
for label = 1 : numel(unique_labels)
    indices = train_labels ~= label;
    candidate_labels = setdiff(unique_labels, label);
    tmp(indices) = randsample(candidate_labels, sum(indices), true);
end

train_y = ind2vec(train_labels).';
train_z = ind2vec(tmp).';
clear tmp;

tmp = zeros(size(test_labels));
for label = 1 : numel(unique_labels)
    indices = test_labels ~= label;
    candidate_labels = setdiff(unique_labels, label);
    tmp(indices) = randsample(candidate_labels, sum(indices), true);
end

test_y = ind2vec(test_labels).';
test_z = ind2vec(tmp).';
clear tmp;

%% train models
theta_ours = our_method_v1(train_DT, train_y, train_DT, train_z, unique_labels);
theta_all = zeros(numel(unique_labels), size(train_DT, 2));
for i = 1 : numel(unique_labels)
    ind = find(train_labels == unique_labels(i));
    theta_all(i,:) = sum(train_DT(ind, :));
end
for i = 1 : numel(unique_labels)
    theta_all(i,:) = theta_all(i,:) / sum(theta_all(i,:));
end

%% compare performance
accu_all = get_accuracy(theta_all, test_DT, test_labels.');
accu_ours = get_accuracy(theta_ours, test_DT, test_labels.');

%% plot
figure();
hold on
plot(accu_all,'*');
plot(accu_ours,'o');
mean_all = refline([0, mean(accu_all)]);
mean_ours = refline([0, mean(accu_ours)]);
mean_all.Color = 'g';
mean_ours.Color = 'm';
legend('naive bayes','our method', 'avg-acc (naive bayes)', 'avg-acc (our method)');
title('Comparison between our proposed method and Naive Bayes method');








