function [accuracy] = get_accuracy(theta,test_feature,test_label)
%GET_ACCURACY return accuracy of test data using centroid theta

dif_label = unique(test_label);
[n_samples] = size(test_feature,1);
accuracy = zeros(size(dif_label,1)+1,1);    % the last entry is total accuracy

estimator = zeros(n_samples,1);
for i = 1:n_samples
%     likelihood = sum(log(theta).*test_feature(i,:),2);
    likelihood = log(theta) * test_feature(i,:).';
%     plot(likelihood)
%     waitforbuttonpress
    [~,ind] = max(likelihood);
    estimator(i) = dif_label(ind);
end

for i = 1:size(dif_label,1)
    ind = find(test_label == dif_label(i));
    res = estimator(ind) - test_label(ind);
    accuracy(i) = sum(res==0)/size(ind,1);
end

res = estimator - test_label;
accuracy(end) = sum(res==0)/n_samples;

end

