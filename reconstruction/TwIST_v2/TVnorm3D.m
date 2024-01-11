function y = TVnorm3D(x, w, h, nz)

y = 0;

for j = 1:nz
    xz = x((j-1)*w*h+1 : j*w*h);
    xz = reshape(xz, h, w);
    y = y + sum(sum(sqrt(diffh(xz).^2+diffv(xz).^2)));
end

