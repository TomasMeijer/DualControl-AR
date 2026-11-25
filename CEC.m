%% DUAL CONTROL REVIEW CASE STUDY: CEC.m
%% Verification
% For Gaussian noise: rescale to achieve unit variance
if w_var ~= 0
    Sigma = w_var^2*eye(n);
    T1 = sqrt(Sigma);
else
    T1 = eye(n);
end

Alocal = T1\A*T1;
Blocal = T1\B;
Qlocal = T1'*Qlocal*T1;
x0local = T1\x0;
wk_local = T1 \ wk;

% Test cost function
if Rlocal ~= 1
    error('Input cost R should be one.');
elseif min(eig(Qlocal)) <= 1
    error('State cost Q should be larger than identity.')
end

%% Simulate
xk = [x0local zeros(n,Lsim)];
uk = zeros(1,Lsim);
Bhat = zeros(n,Lsim);
phase = zeros(1,Lsim);
lk = 0;
Kk = 0;
Khat = zeros(Lsim,n);
xhist = 0;
uhist = 0;
gamma_cec = 1;
for k = 1:Lsim
    xhist = mu*xhist + norm(xk(:,k));
    if xhist > w_var^2*n*(k-1)^(1+gamma_cec) 
        lk = 0;
    elseif xhist < w_var^2*n*(k-1)^(1+gamma_cec/2) 
        lk = 1;
    else
        % Do nothing (i.e., lk doesn't change)
    end

    % Least-square estimation
    XU = zeros(n,1);
    UU = 0;
    for s = 1:k-1
        XU = mu*XU + (xk(:,s+1)-Alocal*xk(:,s))*uk(:,s)';
        UU = mu*UU + uk(:,s)*uk(:,s)';
    end
    Bhat(:,k) = XU*pinv(UU);
    
    % Compute CE
    [Pk,Kk] = idare(Alocal,Bhat(:,k),Qlocal,Rlocal);
    Kk = -Kk;
    Khat(k,:) = Kk;
    
    if (lk == 1) && (norm(Kk)^2 <= (k-1)^gamma_cec) && (min(eig(sum(uk.*uk))) >= sqrt(k-1)) % Exploitation
        uk(:,k) = Kk*xk(:,k); 
    else % Exploration
        uk(:,k) = randn(1,1); 
        phase(k) = 1;
    end
    
    % Simulate dynamics
    xk(:,k+1) = Alocal*xk(:,k) + Blocal*uk(:,k) + wk_local(:,k);
    uhist = uhist + uk(:,k)*uk(:,k)';
end


% Scale back states and estimated Bhat
Bhat = T1*Bhat;
Khat = Khat*inv(T1);
xk = T1*xk;