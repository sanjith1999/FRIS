function demo_FRIS_l2_TV
%
%     xe = arg min 0.5*||A x-y||^2 + tau TV(x)
%             x
%

clear all; close all; clc

% load the data
data = load('E:\Aca\FYP\codes\FRIS _Copy\reconstruction\data\3DS_m3_np2_df1_32x32x5.mat');

m = 3; % no of patterns
np = 2; % no of detected planes
df = 1; % downsampling factor
h = 32; % height of a plane (dy = 1/3 um)
w = 32; % width of a plane (dx = 1/3 um)
nz = 5; % number of planes (dz = 2 um)


% Measurement matrix
R = data.A;
% figure;
% % Loop through each pattern
% for i = 1:m
%     currentMatrix = R_all((i-1)*w*df*h*df+1:i*w*df*h*df , :);
%     subplot(m, 1, i);
%     imagesc(currentMatrix);
%     axis off;
%     colormap gray;
% end
% sgtitle('Matrices for Each Pattern');


x = data.X_original'; % Original object

% detected image (clean)
y = data.Y;
% visualize detected images
for p = 1:np
    figure;
    numRows = ceil(sqrt(m));
    numCols = ceil(m / numRows);
    step = w*df*h*df;
    j = 1;
    for i = p:np:m*np
        subplot(numRows, numCols, j);
        curIm = reshape(y((i-1)*step + 1 : i*step), w*df, h*df);
        imagesc(curIm);
        colormap gray;
        title(['For pattern : ' num2str(j)]);
        j = j + 1;
    end
    sgtitle(['Noiseless Image of plane number ' num2str(p)]); 
end


% detected image (noisy)
y = data.Yn;
% visualize detected images
for p = 1:np
    figure;
    numRows = ceil(sqrt(m));
    numCols = ceil(m / numRows);
    step = w*df*h*df;
    j = 1;
    for i = p:np:m*np
        subplot(numRows, numCols, j);
        curIm = reshape(y((i-1)*step + 1 : i*step), w*df, h*df);
        imagesc(curIm);
        colormap gray;
        title(['For pattern : ' num2str(j)]);
        j = j + 1;
    end
    sgtitle(['Noisy Image of plane number ' num2str(p)]); 
end

% remove mean (not necessary)
% mu=mean(x);
% x=x-mu;

%TwIST handlers
% Linear operator handlers
A = @(x) R*reshape(x, w*h*nz, 1);
AT = @(y) reshape(R'*y, w*h*nz, 1);

% observed data
y = data.Yn;

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

rows = 2;
cols = nz;
figure;
for j = 1:nz
    xz = x((j-1)*w*h+1 : j*w*h);
    xz = reshape(xz, h, w);
    
    x_twist_z = x_twist((j-1)*w*h+1 : j*w*h);
    x_twist_z = reshape(x_twist_z, h, w);
    
    subplot(rows, cols, j);
    colormap gray;
    imagesc(xz);
    title(sprintf('Plane: %d', j));
    
    subplot(rows, cols, j + cols);
    colormap gray;
    imagesc(x_twist_z);
end
sgtitle('Comparison');


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
fprintf(1, 'MSE Loss: %.4e\n', mse_twist(end));