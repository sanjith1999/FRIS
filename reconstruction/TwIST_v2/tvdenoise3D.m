function u3D = tvdenoise3D(f,lambda,iters,w,h,nz)

if nargin < 6
    Tol = 1e-2;
end

if lambda < 0
    error('Parameter lambda must be nonnegative.');
end

dt = 0.25;

u3D = zeros(h*w*nz, 1);

% apply denoising plane by plane
for j = 1:nz
    fz = f((j-1)*w*h+1 : j*w*h);
    fz = reshape(fz, h, w);
    N = size(fz);
    id = [2:N(1),N(1)];
    iu = [1,1:N(1)-1];
    ir = [2:N(2),N(2)];
    il = [1,1:N(2)-1];
    p1 = zeros(size(fz));
    p2 = zeros(size(fz));
    divp = zeros(size(fz));
    lastdivp = ones(size(fz));

    if length(N) == 2           % TV denoising
        %while norm(divp(:) - lastdivp(:),inf) > Tol
        for i=1:iters
            lastdivp = divp;
            z = divp - fz*lambda;
            z1 = z(:,ir) - z;
            z2 = z(id,:) - z;
            denom = 1 + dt*sqrt(z1.^2 + z2.^2);
            p1 = (p1 + dt*z1)./denom;
            p2 = (p2 + dt*z2)./denom;
            divp = p1 - p1(:,il) + p2 - p2(iu,:);
        end
    elseif length(N) == 3       % Vectorial TV denoising
        repchannel = ones(N(3),1);

        %while norm(divp(:) - lastdivp(:),inf) > Tol
        for i=1:iters
            lastdivp = divp;
            z = divp - fz*lambda;
            z1 = z(:,ir,:) - z;
            z2 = z(id,:,:) - z;
            denom = 1 + dt*sqrt(sum(z1.^2 + z2.^2,3));
            denom = denom(:,:,repchannel);
            p1 = (p1 + dt*z1)./denom;
            p2 = (p2 + dt*z2)./denom;
            divp = p1 - p1(:,il,:) + p2 - p2(iu,:,:);
        end
    end

    uz = fz - divp/lambda;
    uz = reshape(uz, w*h, 1);
    u3D((j-1)*w*h+1 : j*w*h) = uz;

end


