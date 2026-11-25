%% DUAL CONTROL REVIEW CASE STUDY: wRLS_CE_LQR.m
% Randomization parameter
eps_str = 1/(4.1*n); % Strictly between 0 and 1/4n

% Hysteresis constant for the self-convergent controllability search;
% must satisfy 1 >= 2*gamma_str + gamma_str^2 (paper eq. 35).
% gamma_str = 1/2 violates this (2*0.5+0.5^2 = 1.25 > 1), so it's fixed to 0.4.
gamma_str = 0.4;

%% Verification
if norm(Qlocal-eye(n)) ~= 0
    error('State cost should be normalized to identity.');
end

%% Simulate
xk = [x0 zeros(n,Lsim)];
uk = zeros(1,Lsim);
Sk = zeros(n,n);

% M(theta) is the Sylvester resultant of the KNOWN A(z) and the ESTIMATED
% B(z) (paper eq. 32); 
M = @(theta) [toeplitz([1;a(:);zeros(n-1,1)],[1,zeros(1,n-1)]), ...
              toeplitz([0;theta(1:n);zeros(n-1,1)],zeros(1,n))];

theta_k = zeros(n+2,1);
P0 = 0.5*eye(n+2);
Pk = P0;
Khat = zeros(Lsim,n);
Bhat = zeros(n,Lsim);

what_km1 = zeros(n,1); % what_{k-1}
what_km2 = zeros(n,1); % what_{k-2}
Rsum = 0;              % running sum_i ||Phi_i||_F^2, for the weight sequence (eq. 14)
phase = zeros(1,Lsim);

for k = 1:Lsim
    % Compute theta_hat_k
    Pk_sqrt = sqrtm(Pk);
    eta_k = rand(n+2,1);
    if k == 1
        beta_k = eta_k;
    elseif norm(det(M(theta_k+Pk_sqrt*eta_k))) >= (1+gamma_str)*norm(det(M(theta_k+Pk_sqrt*beta_k)))
        beta_k = eta_k;
    else
        % Do nothing: beta_k = beta_{k-1}
    end
    theta_hat_k = theta_k + Pk_sqrt*beta_k;

    % Extract Bhat_k (only the first n entries of theta_hat_k are B)
    Bhat(:,k) = theta_hat_k(1:n);

    % Compute Sk
    Sk = A'*Sk*A - A'*Sk*Bhat(:,k)*((Rlocal+Bhat(:,k)'*Sk*Bhat(:,k))\(Bhat(:,k)'*Sk*A)) + eye(n);

    % Compute Kk
    Kk = -(Rlocal+Bhat(:,k)'*Sk*Bhat(:,k))\Bhat(:,k)'*Sk*A;

    % Compute control input
    Khat(k,:) = Kk;
    uk(:,k) = Kk*xk(:,k) + randn(1,1)/(max((k-1),1)^(eps_str/2));
    phase(k) = 1/(max((k-1),1)^(eps_str/2));

    % Simulate dynamics
    xk(:,k+1) = A*xk(:,k) + B*uk(:,k) + wk(:,k);

    % Compute next theta_k = [b;c0;c1] via extended least squares (eqs. 9-11),
    % regressing the state innovation on u_k AND the previous two estimated
    % disturbance vectors, and form what_k (the analogue of eq. 13's what_t)
    Phi_k = [uk(:,k)*eye(n), what_km1, what_km2];         % n x (n+2)
    what_k = (xk(:,k+1) - A*xk(:,k)) - Phi_k*theta_k;      % uses theta_k as of BEFORE this update

    Rsum = Rsum + norm(Phi_k,'fro')^2;
    r_k = norm(inv(P0)) + Rsum;
    alpha_k = 1/log(r_k)^1.5;
    
    % Recursive WLS (eq. 9-13)
    Sk_phi = (1/alpha_k)*eye(n) + Phi_k*Pk*Phi_k';
    Lk = Pk*Phi_k'/Sk_phi;
    theta_k = theta_k + Lk*what_k;
    Pk = Pk - Lk*Phi_k*Pk;
    Pk = (Pk+Pk')/2; % symmetrize numerically

    % Shift the estimated-disturbance history
    what_km2 = what_km1;
    what_km1 = what_k;
end
