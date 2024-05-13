function P = updata_p(Y, Q, lambda, cmm, pp)

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

ppp = repmat(pp,m,1);
eta = (Q - ppp).*Y;
eta = reshape(eta',m*q,1);


Q = reshape(Q',m*q,1);
cmm = repmat(cmm,m,1);
f = -2*Q - lambda*cmm(:,1).*eta;
Outputs = quadprog(H, f, [], [], A, beq, lb, ub, [], opts);
P = reshape(Outputs,q,m);
P = P';

end

