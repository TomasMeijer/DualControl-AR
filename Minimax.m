%% DUAL CONTROL REVIEW CASE STUDY: Minimax.m
% Initial covariance matrix
Z0 = zeros(2*n+1,2*n+1);

q = 1-gamma^(-2); 


%% Verification 
% Verify that set of admissible input matrices (Bcal+) is nonempty
[U,D] = eig(A'*A-(1-gamma^(-2))*(eye(n)-Qlocal));
[lambda,idx] = sort(diag(D),'descend');
if n>=2
    if lambda(2)>0
        error('Set B is empty: lambda2 > 0.');
    end
end
U = U(:,idx);
U1 = U(:,1);

% Verify that gamma is sufficiently large
if gamma^2 < (1+2*norm(A)^2/(1-gamma^(-2)))*max(1+Rlocal+beta^2,norm(inv(Qlocal)))
    error('Invalid choice for gamma.');
end

% Check whether the true B satisfies the Ricatti inequality
d = 1-gamma^(-2);
if min(eig(eye(2)-Q-A'*A/d + A'*B*inv(d*R+beta)*B'*A/d)) < 1e-9
    error('The true B does not satisfy the Ricatti inequality.')
end

%% Simulate
xk = [x0 zeros(n,Lsim)];
uk = zeros(1,Lsim);
Bhat = zeros(n,Lsim);
phase = zeros(1,Lsim);
Zk = Z0;
Bbhatk = zeros(n+1,Lsim);
sigmas = zeros(1,Lsim);

p = struct('worst_case_method','geometry','main_solver','scalar','A',A,'x',zeros(n,1),'S',Qlocal,'R',Rlocal,'gamma',gamma,'beta',beta,'Z',zeros(2*n+1,2*n+1),'lambda',lambda,'U',U);
for k = 1:Lsim
    p.x = xk(:,k);
    p.Z = Zk;

    out = minimax_solver(p);

    sigma2 = out.variance_u;
    if sigma2 < 0
        error('Negative variance');
    end
    sigmas(k) = sqrt(sigma2);
    uk(:,k) = out.mean_u + sqrt(sigma2)*randn(1 ,1);
    if sigma2 > 0
        phase(k) = 1;
    end
    Bhat(:,k) = out.worst_B;
    Bbhatk(:,k) = [out.worst_B; out.worst_b];
    
    % Simulate dynamics
    xk(:,k+1) = A*xk(:,k) + B*uk(:,k) + wk(:,k);
    Zk = mu*Zk + [xk(:,k); uk(:,k); -xk(:,k+1)]*[xk(:,k); uk(:,k); -xk(:,k+1)]'; % Compute Z_{k+1}
end