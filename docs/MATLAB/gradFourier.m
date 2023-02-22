function grad = gradFourier(img)

% compute differencing operator in the frequency domain
nx = size(img, 2);
hx = ceil(nx/2)-1;
k = (2i*pi/nx)*(0:hx);     % ik 
k(nx:-1:nx-hx+1) = -k(2:hx+1);  % correct conjugate symmetry

% compute "gradient" in x using fft
gx = ifft2( bsxfun(@times, fft2(img), k') );
gy = ifft2( bsxfun(@times, fft2(img), k) );

grad = sqrt(gx.*gx + gy.*gy);

% figure, imshow(gx, []);      % see result
% figure, imshow(gy, []);      % see result
% 
% gx = gx - min(min(gx));
% gx = gx/max(max(gx));
% gy = gy - min(min(gy));
% gy = gy/max(max(gy));
% 
% imwrite(gx,'gx.png' );
% imwrite(gy,'gy.png' );
end