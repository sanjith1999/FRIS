function demo_FRIS_l2_l1 
% algorithms in  the l2-l1 optimization problem 
%
%     xe = arg min 0.5*||A x-y||^2 + tau ||x||_1
%             x
%
% where A is a generic matrix and ||.||_1 is the l1 norm.

% load the data
data = load('E:\Aca\FYP\codes\twist\data\neural_cell_m_8_x_16x16x8.mat');

% Measurement matrix
R = data.A;
% Original data
x = data.X_original';
% signal length 
n = numel(x);

%TwIST handlers
% Linear operator handlers
hR = @(x) R*x;
hRt = @(x) R'*x;
% define the regularizer and the respective denoising function
% TwIST default
%Psi = @(x,th) soft(x,th);   % denoising function
%Phi = @(x) l1norm(x);       % regularizer

% observed data
y = hR(x);

% regularization parameter 
tau = 0.1*max(abs(hRt(y)));

% TwIST parameters
lambda1 = 1e-4;  

%             If min eigenvalue of A'*A == 0, or unknwon,  
%             set lam1 to a value much smaller than 1. 
%             TwIST is not very sensitive to this parameter
%             
%
%             Rule of Thumb: 
%                 lam1=1e-4 for severyly ill-conditioned problems
%                 lam1=1e-2 for mildly  ill-conditioned problems
%                 lam1=1    for A unitary direct operators



% stopping theshold
tolA = 1e-5;

% -- TwIST ---------------------------
% stop criterium:  the relative change in the objective function 
% falls below 'ToleranceA'
[x_twist,x_debias_twist,obj_twist,...
    times_twist,debias_start_twist,mse]= ...
         TwIST(y,hR,tau, ...
         'Lambda', lambda1, ...
         'Debias',0,...
         'AT', hRt, ... 
         'Monotone',1,...
         'Sparse', 1,...
         'Initialization',0,...
         'StopCriterion',1,...
       	 'ToleranceA',tolA,...
         'Verbose', 1);
   

h_fig = figure(2);
set(h_fig,'Units','characters',...
        'Position',[30 10 150 45])
subplot(2,2,1)
semilogy(times_twist,obj_twist, 'b', 'LineWidth',2)
legend('TwIST')
st=sprintf('\\lambda_1 = %2.1e',tau);
title(st);
xlabel('CPU time (sec)')
ylabel('Obj. function')
grid
subplot(2,2,2)
plot(1:n,x_twist,'b', 1:n, x+2.5,'k','LineWidth',2)
axis([1 n -1, 5]);
legend('TwIST','Original','x')
st=sprintf('TwIST MSE = %2.1e', sum((x_twist-x).^2)/prod(size(x)));

title(st)

fprintf('TwIST  MSE = %2.1e\n', sum((x_twist-x).^2)/prod(size(x)));
fprintf(1,'TwIST   CPU time - %f\n', times_twist(end));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Decrease tau %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  regularization parameter 

tau = 0.00002*max(abs(hRt(y)));
% TwIST parameters

% -- TwIST ---------------------------
% stop criterium:  the relative change in the objective function 
% falls below 'ToleranceA'
[x_twist,x_debias_twist,obj_twist,...
    times_twist,debias_start_twist,mse]= ...
         TwIST(y,hR,tau, ...
         'Debias',0,...
         'AT', hRt, ... 
         'Monotone',1,...
         'Initialization',0,...
         'StopCriterion',1,...
       	 'ToleranceA',tolA,...
         'Verbose', 1);
     


figure(2)
subplot(2,2,3)
semilogy(times_twist,obj_twist, 'b', 'LineWidth',2)
axis([times_twist(1) times_twist(end) obj_twist(end) obj_twist(1)])
legend('TwIST')
st=sprintf('\\lambda_2 = %2.1e',tau);
title(st);
xlabel('CPU time (sec)')
ylabel('Obj. function')
grid

subplot(2,2,4)
plot(1:n,x_twist,'b', 1:n, x+2.5,'k','LineWidth',2)
axis([1 n -1, 5]);
legend('TwIST','Original','x')
st=sprintf('TwIST MSE = %2.1e', sum((x_twist-x).^2)/prod(size(x)));

title(st)

fprintf('TwIST  MSE = %2.1e\n', sum((x_twist-x).^2)/prod(size(x)));
fprintf(1,'TwIST   CPU time : %f\n', times_twist(end));


