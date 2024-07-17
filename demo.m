% 基于代价敏感的候选标签集消歧策略
clear

% personal data
load('DATA\personal_data\lost.mat');

% index partition
load('lost0813idx.mat');
% [tr_idx, te_idx] = data_segment(data);

% data inital
data = data_initial(data, 2);

% optim
mu = 0.04; % overfitting
lambda = 0.6; % C+
max_iter = 30;
c = 0.2;

cl_acc = zeros(10,1);
Precision = zeros(10,1);
Recall = zeros(10,1); 
F_measure = zeros(10,1);
MAUC = zeros(10,1);

for i = 1:10
    fprintf('fold i = %d\n',i)
    train_data = data(tr_idx(:,i),:);
    train_p_target = full(partial_target(:,tr_idx(:,i))');
    train_target = target(:,tr_idx(:,i))';
    test_data = data(te_idx(:,i),:);
    test_target = target(:,te_idx(:,i))';

    [test_outputs, cl_acc(i), Precision(i), Recall(i), F_measure(i), MAUC(i)] = pl_cgr(train_data, train_p_target, test_data, test_target, mu, lambda, c, max_iter);
end

acc = mean(cl_acc);
P = mean(Precision);
R = mean(Recall);
F = mean(F_measure);
M = mean(MAUC);






