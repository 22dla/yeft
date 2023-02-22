clc
clear 
close all
addpath('dft')

I = imread('../images/area_desk/0.png');
I = double(I);

% a = 1;
% b = 10 + pi/2;
% N = 100;
% dx = (b-a)/N;
% x = a + dx*(0:N-1);
% I = sin(x);
% Ix = cos(x);
% 
% % compute differencing operator in the frequency domain
% [ny, nx] = size(I);
% 
% ftx = 2*pi*1i/(b-a)*[0:nx/2-1 0 -nx/2+1:-1];     % ik (x)
% 
% % compute "gradient" in x using fft
% F = fftn(I);
% gx = ifftn( ftx.*F );
% 
% Nx = size(x,2);
% k = 2*pi/(b-a)*[0:Nx/2-1 0 -Nx/2+1:-1];
% dFdx = ifft(1i*k.*fft(I));
% plot(x, dFdx);      % see result
% 
% figure, plot(Ix);      % see result

% domain
a = 1;
b = 1 + pi/2;
N = 1000;
T = (b-a)/N;
Fs = 1/T;
x = linspace(a,b,N);
x = a + T*(0:N-1);
omega = Fs/2*linspace(0,1,N);

% function
w = 10;
f = sin(w*x).^2;

% exact derivatives
dfdx = 2*w*sin(w*x).*cos(w*x);

% fourier derivatives
nx = size(x,2);
hx = ceil(nx/2)-1;
k = (2*pi/(b-a))*(0:nx-1);     % ik 
k(nx:-1:nx-hx+1) = -k(2:hx+1);
% 
% k = 2*pi/(b-a)*[0:nx/2-1 0 -nx/2+1:-1];

F = dft1D(f);
dFdx = idft1D(1i*k.*F);

% graph result
figure;
plot(x,dfdx,'r-',x,dFdx,'k:','LineWidth',2);
legend('df/dx','Fourier df/dx');

