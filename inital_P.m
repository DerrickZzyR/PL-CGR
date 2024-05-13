function [P] = inital_P(Y, Q, c)

[m, q] = size(Q);
opts = optimoptions('quadprog',...
    'Algorithm','interior-point-convex','Display','off');
Aeq = ones(1,q);
A = repmat({Aeq},1,m);
A = spblkdiag(A{:});
beq = ones(m,1);
lb = sparse(m*q,1);
ub = reshape(Y',m*q,1);
h = 2*speye(q, q);
H = repmat({h},1,m);
H = spblkdiag(H{:});

[m, l] = size(Q);
P = zeros(m, l);
% lambda = 0.2;

YQ = Y.*Q;
[~, idx] = max(YQ, [], 2);
index = zeros(m,q);
for i = 1:m
    index(i,idx(i)) = 1;
end

atp = reshape(index',m*q,1);
Q = reshape(Q',m*q,1);
f = -2*Q - c*atp;
Outputs = quadprog(H, f, [], [], A, beq, lb, ub, [], opts);
P = reshape(Outputs,q,m)';

end