%% DUAL CONTROL REVIEW CASE STUDY: Example 1 - Nonminimum phase system with Gaussian white noise
% Initialize
clc; clear all; close all;
projectRoot = fileparts(mfilename('fullpath'));
addpath(genpath([projectRoot,'/dependencies']));
options = optimoptions('fmincon','Display','none');

set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');

preserve = 1; % Don't delete data from the individual scripts

% rng(29102026); % Fix random noise (uncomment for reproducibility)

% Parameters
n = 2; % System dimension (cannot be changed for coloured noise)
Lsim = 1e2; % Simulation length
x0 = 15*randn(n,1); % Initial state
mu = 0.98; % Forgetting factor

%% Model: n-th order ARMAX model
% y(t) = -a(1)y(t-1) - ... - a(n)y(t-n) + b(1)u(t-1) + ... + b(n)u(t-n) + e(t) + c(1)e(t-1) + ... + c(n)e(t-n)
a = [0.5 0.5]; 
b = [1 3/2]; % Non-minimum phase

% Original systemge
A0 = [-a' [eye(n-1); zeros(1,n-1)]];
B0 = b'; 

% Transform the system such that sigma(A) < 1
% T = chol(dlyap(A0,eye(n)),'lower');
T = diag([1,1]); % Currently does nothing
A = T\A0*T;
B = T\B0;

%% Generate noise
wk = zeros(n,Lsim);
w_var = 1;
for k = 1:Lsim
    wk(:,k) = w_var*randn(n,1);
end

%% Cost function (original)
Q = 0.1*eye(n);
R = 0.01;

%% Simulate the different control schemes
controllers = {'wRLS CE-LQR','CEC','Minimax'};
titles = {'Weighted recursive least squares','Regret rate minimization','Minimax approach'};

xk_i = cell(size(controllers)); % Store state trajectories
uk_i = cell(size(controllers)); % Store control actions
Bhat_i = cell(size(controllers)); % Store estimated B
phase_i = cell(size(controllers)); % Store exploration/exploitation phase
Khat_i = cell(size(controllers)); % Store Khat

% Weighted recursive least squares
% Normalize Q by scaling cost function
Qlocal = 1*eye(n);
Rlocal = R/(Q(1,1));

run("wRLS_CE_LQR.m");
xk_i{1} = xk;
uk_i{1} = uk; 
Bhat_i{1} = Bhat;
phase_i{1} = phase;

% Regret rate minimization
% Normalize R by scaling cost function
Qlocal = Q/R;
Rlocal = 1;

run("CEC.m");
xk_i{2} = xk;
uk_i{2} = uk;
Bhat_i{2} = Bhat;
phase_i{2} = phase;

% Minimax approach
Qlocal = Q;
Rlocal = R;
beta = 4;
gamma = 15;

run("Minimax.m");
xk_i{3} = xk;
uk_i{3} = uk;
Bhat_i{3} = Bhat;
phase_i{3} = phase;


%% Visualize results
cost_i = cell(length(controllers),1);
for i = 1:length(controllers)
    xk = xk_i{i};
    uk = uk_i{i};
    cost = zeros(Lsim,1);
    for k = 1:Lsim
        if k == 1
            cost(k) = xk(:,k)'*Q*xk(:,k) + uk(:,k)'*R*uk(:,k);
        else
            cost(k) = cost(k-1) + xk(:,k)'*Q*xk(:,k) + uk(:,k)'*R*uk(:,k);
        end
    end
    cost_i{i} = cost;
end

disp('Achieved cost:');
tend = Lsim/2;
for i = 1:length(controllers)
    display([controllers{i},': ',num2str(cost_i{i}(tend))]);
end

% Easy-to-see, colorblind-friendly plot colors
colors = [
    0.9290  0.6940  0.1250   % yellow
    0.8500  0.3250  0.0980   % orange
    0.0000  0.4470  0.7410   % blue
    0.4940  0.1840  0.5560   % purple
    0.4660  0.6740  0.1880   % green
    0.3010  0.7450  0.9330   % cyan
    0.6350  0.0780  0.1840   % dark red
];

% Plot results
k = 0:Lsim;
figure(100);
% One panel per method
for i = 1:length(controllers)
    ax = subplot(length(controllers),1,i);

    % Plot \|x(t)\|
    yyaxis left
    h1 = plot(k,vecnorm(xk_i{i}),'LineWidth',2,'Color',colors(i,:));
    hold on;

    % Plot u(t)
    h2 = stairs(k(1:end-1),abs(uk_i{i}),'--','LineWidth',2,'Color',colors(i,:));
    ylim([0,30]);
    ylabel('$\|x(t)\|,~|u(t)|$');

    % Map phase in [0.75, 1] to alpha in [alphaMin, alphaMax]; phase = 0 stays transparent
    alphaMin = 0.1;   % alpha at phase = 0.75 (faintest visible shading)
    alphaMax = 0.35;   % alpha at phase = 1
    lo = 0.75;
    ph = phase_i{i}(:);
    n  = numel(ph);
    a  = zeros(n,1);
    on = ph > 0;
    s  = min(max((ph(on) - lo) / (1 - lo), 0), 1);   % normalize to [0,1], clipped
    a(on) = alphaMin + (alphaMax - alphaMin) * s;
    t0 = k(1:n);  t1 = k(2:n+1);
    xs = [t0; t1; t1; t0];
    ys = repmat([0; 0; 30; 30], 1, n);
    for j = find(on).'
        p = patch([t0(j) t1(j) t1(j) t0(j)], [0 0 30 30], colors(i,:), ...
            'EdgeColor','none','FaceAlpha',a(j),'HandleVisibility','off');
        uistack(p,'bottom');
    end

    % Plot estimates
    yyaxis right
    h3 = plot(k(1:end-1),Bhat_i{i},':','LineWidth',2,'Color',colors(i,:));
    hold on;
    h4 = yline(B,'Color',[0.5,0.5,0.5],'Linewidth',1.5);
    ylim([-1,4]);
    ax.YAxis(1).Color = [0 0 0];
    ax.YAxis(2).Color = [0 0 0];
    xlim([0,60]);
    ylabel('$B,~\hat{B}_t$');

    if(i~=length(controllers))
        ax.XTickLabel = [];
    else
        xlabel('t');
    end

    % Title, left-aligned
    tt = title(titles(i));
    tt.Units = 'normalized';
    tt.HorizontalAlignment = 'left';
    tt.Position(1) = 0;

    % Horizontal legend on top of the panel, right-aligned with the axes
    hs = gobjects(4,1);
    hs(1) = h1(1);
    hs(2) = h2(1);
    hs(3) = h3(1);
    hs(4) = h4(1);
    lgd = legend(ax, hs, ...
        {'$\|x(t)\|$','$|u(t)|$','$\hat{B}_t$','$B$'}, ...
        'Orientation','horizontal','Interpreter','latex', ...
        'Box','off','Location','none');
    lgd.Units = 'normalized';
    lgd.Position(1) = ax.Position(1) + ax.Position(3) - lgd.Position(3);
    lgd.Position(2) = ax.Position(2) + ax.Position(4) + 0.002;
end