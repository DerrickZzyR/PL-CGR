function [Qt, cl_acc, Precision, Recall, F_measure, MAUC] = pl_cgr(train_data, train_p_target, test_data, test_target, mu, lambda, c, max_iter)

% inital P
ker = 'rbf';
par = mean(pdist(train_data));
[Q, ~] = kernelRidgeRegression(train_data, train_p_target, test_data, mu, par, ker);
[P] = inital_P(train_p_target, Q, c);

% inital label threshold, cost
fprintf('Generate label threshold and cost\n');
ratio = 0.1;
[A,lt] = pl_cost(train_data, train_p_target, P, ratio);

step = max_iter/100;
count = 0;
steps = 100/max_iter;
fprintf('Iterative optimization\n')
for j = 1:max_iter
    if rem(j,step) < 1
        fprintf(repmat('\b',1,count-1));
        count = fprintf(1,'>%d%%',round(j*steps));
    end
    %updata classifier
    [Q,~] = kernelRidgeRegression(train_data, P, test_data, mu, par, ker);
    %updata label distrubtion
    [P] = updata_p(train_p_target, Q, lambda, A, lt');
end
fprintf('\n');

%% test acc
[~,Qt] = kernelRidgeRegression(train_data, P, test_data, mu, par, ker);
cl_acc = CalAccuracy(Qt, test_target);
[Precision, Recall, F_measure, MAUC] = imbalance_loss(Qt, test_target, size(test_target,1), size(test_target,2));

end
