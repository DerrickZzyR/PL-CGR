function [Precision, Recall, F, MACU] = imbalance_loss(predict_label, test_target, num_data, num_class)
% precision = TP / (TP+FP)
% recall = TP / (TP+FN)
% specificity = FN / (TN+FP)
% F = (2*recall*precision) / (recall+precision)

% train
% pseudo_train_target = zeros(num_train, num_class);
% [~, tr_ind1] = max(train_outputs, [], 2);
% [~, tr_ind2] = max(train_target, [], 2);
% for i = 1:num_train
%     pseudo_train_target(i,tr_ind1(i)) = 1;
% end

pseudo_test_target = zeros(num_data, num_class);

[~, te_ind1] = max(predict_label, [], 2);
% [~, te_ind2] = max(test_target, [], 2);

for i = 1:num_data
    pseudo_test_target(i, te_ind1(i)) = 1;
end
% tg = full(sum(target, 2))';
% 
% tg = tg >0;

% acc = zeros(1, num_class);
p = zeros(1, num_class);
r = zeros(1, num_class);
f = zeros(1, num_class);
% m = zeros(1, num_class);

for i = 1:num_class
    %     TP_tr = pseudo_train_target(:,i) .* train_target(:,i);
    TP_te = pseudo_test_target(:,i) .* test_target(:,i);
    %     TP = sum(TP_tr)+sum(TP_te);
    FP_te = pseudo_test_target(:,i) .* ~test_target(:,i);
    FN_te = test_target(:,i) .* ~pseudo_test_target(:,i);
    TN_te = ~test_target(:,i) .* ~pseudo_test_target(:,i);
    TP = sum(TP_te);
    FP = sum(FP_te);
    FN = sum(FN_te);
    TN = sum(TN_te);

%     acc(i) = (TP + TN) / (TP + FN + FP + FN);
    if TP == 0
        p(i) = 0;
        r(i) = 0;
    else
        p(i) = TP / (TP + FP);
        r(i) = TP / (TP + FN);
    end

    if p(i) == 0 & r(i) == 0
        f(i) = 0;
    else
        f(i) = 2*p(i)*r(i) / (p(i)+r(i));
    end
%     [~,~,m(i)] = calculate_roc(predict_label(:,i), test_target(:,i));
end
Precision = mean(p);
Recall = mean(r);
F = mean(f);

% Precision = mean(p(tg));
% Recall = mean(r(tg));
% F = mean(f(tg));

% MACU = calMAUC(test_target', pseudo_test_target', predict_label);
MACU = calMAUC(test_target', pseudo_test_target', pseudo_test_target);
