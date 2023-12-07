function demo_FRIS_l2_TV 
% algorithms in  the l2-l1 optimization problem 
%
%     xe = arg min 0.5*||A x-y||^2 + tau TV(x)
%             x
%

% load the data
data = load('E:\Aca\FYP\codes\twist\data\neural_cell_m_4_x_16x16x8.mat');

% width of a plane
w = 16;

total_mse = 0;

% iterate through planes
for z = 1:8
    % Measurement matrix
    R = data.A;
    % for a plane
    R = R(:, (z-1)*w^2+1: z*w^2);

    % Original data
    x = data.X_original';
    % get a plane
    x = x((z-1)*w^2+1 : z*w^2);

    % remove mean (not necessary)
    mu=mean(x);
    x=x-mu;

    %TwIST handlers
    % Linear operator handlers
    A = @(x) R*reshape(x, w*w, 1);
    AT = @(x) reshape(R'*x, w, w);

    % observed data
    y = A(x);

    % denoising function;
    % tv_iters = 5;
    % Psi = @(x,th)  tvdenoise(x, 2/th, tv_iters);
    % TV regularizer;
    Phi = @(x) TVnorm(x);

    % regularization parameter 
    absAty = abs(AT(y));
    tau = 0.001*max(absAty(:));
    % extreme eigenvalues (TwIST parameter)
    lam1=1e-4;   

    % initialization 
    x0 = AT(y);

    % stopping theshold
    tolA = 1e-5;

    % -- TwIST ---------------------------
    % stop criterium:  the relative change in the objective function 
    % falls below 'ToleranceA'
    [x_twist,dummy,obj_twist,...
        times_twist,dummy,mse_twist]= ...
             TwIST(y,A,tau,...
             'AT', AT, ...
             'lambda',lam1,...
             'True_x', reshape(x, w, w),...
             'Phi',Phi, ...
             'Monotone',1,...
             'Initialization',x0,...
             'StopCriterion',1,...
             'ToleranceA',tolA,...
             'Verbose', 1);


    figure;
    subplot(2, 2, 1);
    colormap gray;
    imagesc(reshape(x, w, w));
    axis off;
    title('Original image');

    subplot(2, 2, 2);
    colormap gray;
    imagesc(reshape(x_twist, w, w));
    axis off;
    title('TwIST restored image');
    sgtitle(sprintf('Plane: %d, MSE Loss: %.4e', z, mse_twist(end)));
    drawnow;

    subplot(2,2,3)
    semilogy(times_twist,obj_twist,'r','LineWidth',2)
    legend('TwIST')
    ylabel('Obj. function')
    xlabel('CPU time (sec)')

    grid
    subplot(2,2,4)
    plot(times_twist(2:end),mse_twist(2:end),'r','LineWidth',2)
    legend('TwIST')
    ylabel('MSE')
    xlabel('CPU time (sec)')


    fprintf(1,'TwIST   CPU time : %f\n', times_twist(end));
    
    total_mse = total_mse + mse_twist(end);
end

fprintf(1,'Total mse : %e\n', total_mse);



