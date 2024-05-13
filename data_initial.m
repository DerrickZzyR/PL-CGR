function data = data_initial(data, preprocess)
% data initialization

% preprocess = 3;
if preprocess == 1
    [n,~] = size(data);
    ma = max(data);
    mi = min(data);
    data = (data - repmat(mi, n, 1))./(repmat(ma-mi, n, 1)+1e-6);
    data = 2*data./sum(data, 2);
elseif preprocess == 2
    data = data +eps;
    data = data./repmat(sqrt(sum(data.^2,2)), 1, size(data, 2));
elseif preprocess == 3
    data = zscore(data);
elseif preprocess == 4
    data = data./vecnorm(data, 2, 2);
end