function out = minimax_solver(p)
    A = p.A;
    x = p.x;
    n = size(A,1);
    d = 1-p.gamma^(-2);
    S = p.S;
    beta = p.beta;
    R = p.R;

    L = (norm(A*x)^2-d*x'*(eye(n)-S)*x)/(beta+d*R);
    smin = max(0,L);

    % Optimize only the mean; the second moment is determined exactly.
    cost = @(m) min_u_cost(p,[m;max(m^2,smin)]);

    M = max(sqrt(smin), ...
        sqrt(beta)*norm(A*x)/(1+d*R));

    % Include endpoints, zero, and moment breakpoints.
    candidates = unique([-M;M;0;-sqrt(smin);sqrt(smin)]);

    if M > 0
        [m,~,exitflag] = fminbnd(cost,-M,M, ...
            optimset('Display','none','TolX',1e-9));

        if exitflag ~= 1
            error('Outer scalar search did not converge.');
        end

        candidates(end+1) = m;
    end

    values = arrayfun(cost,candidates);
    [~,idx] = min(values);

    Eu = candidates(idx);
    Eu2 = max(Eu^2,smin);

    out = struct();
    out.mean_u = Eu;
    out.second_moment_u = Eu2;
    out.variance_u = max(0,Eu2-Eu^2);

    [out.worst_B,out.worst_b] = worst_case(p,Eu,Eu2);
end

function val = min_u_cost(p,u_moments)
    A = p.A;
    R = p.R;
    x = p.x;
    d = 1-p.gamma^(-2);

    [~,~,val] = worst_case(p,u_moments(1),u_moments(2));
    val = val + R*u_moments(2) + norm(A*x)^2/d; 
end

function [Bhat,bhat,val] = worst_case(p,Eu,Eu2)
    d = 1-p.gamma^(-2);
    A = p.A;
    n = size(A,1);
    x = p.x;
    gamma = p.gamma;
    Z = p.Z;
    beta = p.beta;

    [Bbhat_plus,val_plus,flag_plus,info_plus] = fmincon(@(Bbhat) -(2*Bbhat(1:n)'*A*x*Eu + Bbhat(end)*Eu2)/d + gamma^2*trace([A'*A A'*Bbhat(1:n) A'; ...
        Bbhat(1:n)'*A Bbhat(end) Bbhat(1:n)'; ...
        A Bbhat(1:n) eye(n)]'*Z),[ones(n,1)/sqrt(n); 1],[zeros(1,n) -1; zeros(1,n) 1],[-1; beta],[],[],[],[],@(Bbhat) nonlincon(Bbhat,p));
    if flag_plus <= 0
        error('Positive internal solve failed: %s',info_plus.message);
    end
    val_plus = -val_plus;
    [Bbhat_minus,val_minus,flag_minus,info_minus] = fmincon(@(Bbhat) -(2*Bbhat(1:n)'*A*x*Eu + Bbhat(end)*Eu2)/d + gamma^2*trace([A'*A A'*Bbhat(1:n) A'; ...
        Bbhat(1:n)'*A Bbhat(end) Bbhat(1:n)'; ...
        A Bbhat(1:n) eye(n)]'*Z),[-ones(n,1)/sqrt(n); 1],[zeros(1,n) -1; zeros(1,n) 1],[-1; beta],[],[],[],[],@(Bbhat) nonlincon([-Bbhat(1:n); Bbhat(end)],p));
    if flag_minus <= 0
        error('Negative internal solve failed: %s',info_minus.message);
    end
    val_minus = -val_minus;
    if(val_plus > val_minus)
        Bhat = Bbhat_plus(1:n);
        bhat = Bbhat_plus(end);
        val = val_plus;
    else
        Bhat = Bbhat_minus(1:n);
        bhat = Bbhat_minus(end);
        val = val_minus;
    end
end

function [c,ceq] = nonlincon(Bbhat,p)
    lambda = p.lambda;
    A = p.A;
    n = size(A,1);
    d = 1-p.gamma^(-2);
    R = p.R;
    beta = p.beta;
    U = p.U;

    ceq = [];
    c = norm(Bbhat(1:n))^2 - Bbhat(end);

    if lambda(2) > 0
        error('Set B is empty: lambda2 > 0.');
    end

    if lambda(1) > 0
        c = [c;
          sqrt(lambda(1)*(beta + d*R - ((Bbhat(1:n)'*A*U(:,2:end)).^2)*lambda(2:end).^(-1)))- Bbhat(1:n)'*A*U(:,1)];
    end
end
