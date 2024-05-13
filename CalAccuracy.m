function [accuracy] = CalAccuracy(outputs, target)
%Calculate the test accuracy for multi-class classification
[~, index1] = max(outputs, [], 2);
[~, index2] = max(target, [], 2);
accuracy = (sum(index1 == index2))/(size(target, 1));
end