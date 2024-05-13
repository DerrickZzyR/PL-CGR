function [tr_idx, te_idx] = data_segment(data)
% data segment

fold = 10;
temp_num = size(data,1);
cv = cvpartition(temp_num,'KFold',fold);
tr_idx = zeros(temp_num,10);
te_idx = zeros(temp_num,10);
for i = 1:fold
    tr_idx(:,i) = cv.training(i);
    te_idx(:,i) = cv.test(i);
end

tr_idx = tr_idx > 0;
te_idx = te_idx > 0;