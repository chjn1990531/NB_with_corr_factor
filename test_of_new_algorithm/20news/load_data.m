function [train_DT, test_DT, train_labels, test_labels] = load_data(train_size)

%% load raw dataset

data = load('data/train.data');
labels = load('data/train.label');

document_indices = data(:, 1);
term_indices = data(:, 2);
freqs = data(:, 3);

n_documents = max(document_indices); % total number of documents
n_terms = max(term_indices);     % total number of terms

% document-term matrix, each entry is term frequency
DT_freq = sparse(document_indices, term_indices, freqs, n_documents, n_terms);

clear data

%% compute tf-idf

idf = log( n_documents ./ sum(DT_freq > 0, 1));

tf_idf = zeros(size(freqs));
for i = 1 : numel(freqs)
    term_index = term_indices(i);
    tf_idf(i) = freqs(i) / idf(term_index);
end

% document-term matrix, each entry is tf-idf
DT_tf_idf = sparse(document_indices, term_indices, tf_idf, n_documents, n_terms);

clear i term_index idf;
%% term selection

% The current document-term matrix contains ~50000 terms, which is too many
% for our task, we only select a subset out of them.

term_importance = full(sum(DT_tf_idf, 1)); % column sum of tf-idf, measuring the term importance
pct_low = prctile(term_importance, 40);    % lower bound percentile
pct_high = prctile(term_importance, 60);   % upper bound percentile

% only term of in between lower and upper bound importance will be selected
selected_term_indices = find((term_importance > pct_low) & (term_importance < pct_high));
n_selected_terms = numel(selected_term_indices);

selected = ismember(term_indices, selected_term_indices);

DT_selected_tf_idf = sparse(document_indices(selected), term_indices(selected), tf_idf(selected), n_documents, n_terms);

%% clear temporary varibles

clear term_importance pct_low pct_high

%% train test split

train_indices = randperm(n_documents, round(train_size * n_documents));
test_indices = setdiff(1 : n_documents, train_indices);

train_DT = DT_selected_tf_idf(train_indices, :);
train_labels = labels(train_indices);
test_DT = DT_selected_tf_idf(test_indices,:);
test_labels = labels(test_indices);

















