function demo_FRIS_l2_TV3D
%
%     xe = arg min 0.5*||A x-y||^2 + tau TV(x)
%             x
%

clear all
close all
clc

% load the data
data = load('E:\Aca\FYP\codes\twist\data\3Dsphere_m16_32x32x11_df2_50dB.mat');

h = 32; % height of a plane (dy = 0.2um)
w = 32; % width of a plane (dx = 0.2um)
nz = 11; % number of planes (dz = 0.5um)
df = 0.5; % downsampling factor
m = 16; % number of patterns

% Measurement matrix
R = data.A;
% figure;
% % Loop through each pattern
% for i = 1:m
%     % Extract the matrix for the i-th pattern
%     currentMatrix = R_all((i-1)*w*df*h*df+1:i*w*df*h*df , :);
%     % Display the current matrix in a subplot
%     subplot(m, 1, i);
%     imagesc(currentMatrix);
%     axis off;
%     colormap gray;
% end
% sgtitle('Visualizing Matrices for Each Pattern');

% Original object
x = data.X_original';

% detected image (clean)
y_all_before_noise = data.Y;
figure;
numRows = ceil(sqrt(m));
numCols = ceil(m / numRows);
for i = 1:m
    subplot(numRows, numCols, i);
    currentImage = reshape(y_all_before_noise((i-1)*w*df*h*df + 1 : i*w*df*h*df), w*df, h*df);
    imagesc(currentImage);
    colormap gray;
    title(['Image ' num2str(i)]);
end
sgtitle('Detected Images - before adding noise');

% detected image (noisy)
y_all = data.Yn;
figure;
numRows = ceil(sqrt(m));
numCols = ceil(m / numRows);
for i = 1:m
    subplot(numRows, numCols, i);
    currentImage = reshape(y_all((i-1)*w*df*h*df + 1 : i*w*df*h*df), w*df, h*df);
    imagesc(currentImage);
    colormap gray;
    title(['Image ' num2str(i)]);
end
sgtitle('Detected Images - noisy');

% remove mean (not necessary)
mu=mean(x);
x=x-mu;

%TwIST handlers
% Linear operator handlers
A = @(x) R*reshape(x, w*h*nz, 1);
AT = @(y) reshape(R'*y, w*h*nz, 1);

% observed data
y = A(x);

% denoising function;
tv_iters = 1;
Psi = @(x,th)  tvdenoise3D(x, 2/th, tv_iters, w, h, nz);
% TV regularizer;
Phi = @(x) TVnorm3D(x, w, h, nz);

% regularization parameter 
absAty = abs(AT(y));
tau = 1e-2*max(absAty(:));
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
         'True_x', x,...
         'Phi',Phi, ...
         'Psi',Psi, ...
         'Monotone',1,...
         'Initialization',0,...
         'StopCriterion',1,...
         'ToleranceA',tolA,...
         'Verbose', 1);

for j = 1:nz
    xz = x((j-1)*w*h+1 : j*w*h);
    xz = reshape(xz, h, w);
    
    x_twist_z = x_twist((j-1)*w*h+1 : j*w*h);
    x_twist_z = reshape(x_twist_z, h, w);
    
    figure
    subplot(1, 2, 1);
    colormap gray;
    imagesc(xz);
    title('Original image');

    subplot(1, 2, 2);
    colormap gray;
    imagesc(x_twist_z);
    title('TwIST restored image');
    sgtitle(sprintf('Plane: %d', j));
    drawnow;
end


figure;
subplot(1,2,1)
semilogy(times_twist,obj_twist,'r','LineWidth',2)
ylabel('Obj. function')
xlabel('CPU time (sec)')

grid
subplot(1,2,2)
plot(times_twist(2:end),mse_twist(2:end),'r','LineWidth',2)
ylabel('MSE')
xlabel('CPU time (sec)')

fprintf(1,'TwIST   CPU time : %f\n', times_twist(end));
fprintf(1, 'MSE Loss: %.4e', mse_twist(end));


