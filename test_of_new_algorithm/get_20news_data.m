clear all
clc

train_feature = load('20news/data/train.data');
test_feature = load('20news/data/test.data');
train_label = load('20news/data/train.label');
test_label = load('20news/data/test.label');


for i = 1:size(test_feature,1)
    test_feature(i,1) = test_feature(i,1) + 11269;
end

feature = [train_feature;test_feature];
label = [train_label;test_label];

clear train_feature test_feature train_label test_label

data = feature;

document_indices = data(:, 1);
term_indices = data(:, 2);
freqs = data(:, 3);

n_documents = max(document_indices); % total number of documents
n_terms = max(term_indices);     % total number of terms

% document-term matrix, each entry is term frequency
feature = sparse(document_indices, term_indices, freqs, n_documents, n_terms);

clear data document_indices term_indices freqs n_documents n_terms i


w = sum(feature);
[~,ind1] = sort(w);
ind1 = ind1(1:100);
ind2 = find(w<100);
ind = setdiff(1:61188,[ind1,ind2]);
feature = feature(:,ind);
feature = full(feature);

save('20_feature','feature')
save('20_label','label')

    