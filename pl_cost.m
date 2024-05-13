function [A,lt] = pl_cost(train_data, train_p_target, Y, ratio)

[~, num_class] = size(train_p_target);

my = zeros(1,num_class);
for i = 1:num_class
    tidx = find(Y(:,i)>0);
    my(i) = mean(Y(tidx,i));
end

idx_p = Y >= my;
k_num = ceil(ratio*sum(idx_p));
k_data = cell(num_class,1);
k_center = cell(num_class,1);
k_idx = cell(num_class,1);
for i = 1:num_class
    k_data{i} = train_data(idx_p(:,i),:);
    % pos and neg
    if (k_num(i) == 0)
        k_center{i} = [];
        k_idx{i} = 0;
    elseif (k_num(i) == 1)
        k_center{i} = k_data{i};
        k_idx{i} = 1;
    else
%         [k_idx{i}, k_center{i}] = kmeans(k_data{i},k_num(i),'EmptyAction','singleton','OnlinePhase','off','MaxIter',500);
        [k_idx{i}, k_center{i}] = kmeans(k_data{i},k_num(i),'EmptyAction','singleton','OnlinePhase','off');
    end
end

cm = zeros(num_class,num_class);
for i = 1:num_class
    for j = 1:num_class
        if ((k_num(i) && k_num(j)) == 0)
            cm(i,j) = 2;
        else
            if i == j
                cm(i,j) = 0;
            else
                cm(i,j) = sum(sum(pdist2(k_data{i}, k_center{j})))/(size(k_data{i},1)*k_num(i));
            end
        end
    end
    tidx = find(cm(i,:) == 2);
    mmax = max(cm(i,:));
    cm(i,:) = (cm(i,:)+0.1)/(mmax+num_class*0.1);
    cm(i,tidx) = 1;
end

e = ones(num_class, num_class) - eye(num_class, num_class);
tmp_lt = cm .* e;
lt = zeros(num_class, 1);
for i = 1:num_class
    FP = sum(tmp_lt(:,i));
    FN = sum(tmp_lt(i,:));
    lt(i) = FP/(FP+FN);
end

cm = ones(num_class, num_class) - cm;
cm = cm .* e;

A = zeros(num_class,2);
for i = 1:num_class
    c01 = sum(cm(i,:));
    c10 = sum(cm(:,i));
    A(i,1) = c01/(c01+c10);
    A(i,2) = c10/(c01+c10);
end

end

