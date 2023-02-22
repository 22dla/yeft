clc
clear 
close all

im = imread('lena.png');
im = rgb2gray(im);
im = double(im);
[n, m] = size(im);

% im = zeros(512, 512);
% im(255:512, :) = 255;
% im( :, 255:512) = 255;

figure, imshow(im)

N = 512;
fourier = zeros(n, m);
for i = 0: (n/N) - 1
    for j = 0: (m/N) - 1
        fourier(1 + i*N:N + i*N, 1 + j*N:N + j*N) = gradFourier(im(1 + i*N:N + i*N, 1 + j*N:N + j*N));
    end
end

figure, imshow(fourier, []);      % see result

gradSobel = imgradient(im, 'Sobel');
figure, imshow(gradSobel, []);      % see result


imwrite(fourier/max(max(fourier)),'gradFourier.png' );
imwrite(gradSobel/max(max(gradSobel)),'gradSobel.png' );