function demo_FRIS_l2_TV 
% algorithms in  the l2-l1 optimization problem 
%
%     xe = arg min 0.5*||A x-y||^2 + tau TV(x)
%             x
%

clear all
close all

% load the data
data = load('E:\Aca\FYP\codes\twist\data\neural_cell_m_16_x_25x25x11_10dB.mat');

% height, width of a plane
h = 25;
w = 25;
% number of planes
nz = 11;
% number of patterns
m = 16;

% Measurement matrix
R_all = data.A;
figure;
% Loop through each pattern
for i = 1:m
    % Extract the matrix for the i-th pattern
    currentMatrix = R_all((i-1)*w*h+1:i*w*h , :);
    % Display the current matrix in a subplot
    subplot(m, 1, i);
    imagesc(currentMatrix);
    colormap gray;
%     title(['Pattern ' num2str(i)]);
end
sgtitle('Visualizing Matrices for Each Pattern');

% Original data
x_all = data.X_original';

% detected image (noisy)
y_all_before_noise = data.Y;
figure;
% Define the number of rows and columns for the subplot grid
numRows = ceil(sqrt(m));
numCols = ceil(m / numRows);
for i = 1:m
    subplot(numRows, numCols, i);
    % Extract the current image from y_all
    currentImage = reshape(y_all_before_noise((i-1)*w*h + 1 : i*w*h), w, h);
    % Display the current image
    imagesc(currentImage);
    colormap gray;
    title(['Image ' num2str(i)]);
end
sgtitle('Detected Images - before adding noise');

% detected image (noisy)
y_all = data.Yn;
figure;
% Define the number of rows and columns for the subplot grid
numRows = ceil(sqrt(m));
numCols = ceil(m / numRows);
for i = 1:m
    subplot(numRows, numCols, i);
    % Extract the current image from y_all
    currentImage = reshape(y_all((i-1)*w*h + 1 : i*w*h), w, h);
    % Display the current image
    imagesc(currentImage);
    colormap gray;
    title(['Image ' num2str(i)]);
end
sgtitle('Detected Images - noisy');

total_mse = 0;

% iterate through planes
for z = 1:nz
    % Measurement matrix for a plane
    R = R_all(:, (z-1)*w*h+1: z*w*h);

    % get a plane
    x = x_all((z-1)*w*h+1 : z*w*h);

    % remove mean (not necessary)
    mu=mean(x);
    x=x-mu;

    %TwIST handlers
    % Linear operator handlers
    A = @(x) R*reshape(x, w*h, 1);
    AT = @(x) reshape(R'*x, w, h);

    % observed data
    y = A(x);

    % denoising function;
    tv_iters = 1;
    Psi = @(x,th)  tvdenoise(x, 2/th, tv_iters);
    % TV regularizer;
    Phi = @(x) TVnorm(x);

    % regularization parameter 
    absAty = abs(AT(y));
    tau = 1e-3*max(absAty(:));
    % extreme eigenvalues (TwIST parameter)
    lam1=1e-4;   

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
             'True_x', reshape(x, w, h),...
             'Phi',Phi, ...
             'Psi',Psi, ...
             'Monotone',1,...
             'Initialization',0,...
             'StopCriterion',1,...
             'ToleranceA',tolA,...
             'Verbose', 1);

    figure;
    subplot(2, 2, 1);
    colormap gray;
    imagesc(reshape(x, w, h));
    axis off;
    title('Original image');

    subplot(2, 2, 2);
    colormap gray;
    imagesc(reshape(x_twist, w, h));
    axis off;
    title('TwIST restored image');
    sgtitle(sprintf('Plane: %d, MSE Loss: %.4e', z, mse_twist(end)));
    drawnow;

    subplot(2,2,3)
    semilogy(times_twist,obj_twist,'r','LineWidth',2)
    ylabel('Obj. function')
    xlabel('CPU time (sec)')

    grid
    subplot(2,2,4)
    plot(times_twist(2:end),mse_twist(2:end),'r','LineWidth',2)
    ylabel('MSE')
    xlabel('CPU time (sec)')

    fprintf(1,'TwIST   CPU time : %f\n', times_twist(end));
    total_mse = total_mse + mse_twist(end);
end

fprintf(1,'Total mse : %e\n', total_mse);



