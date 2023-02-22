function b = dct1D( a )
% a - input vector
% b - result of dct(x)

if size(a, 1) == 1
    N = size(a, 2);
    b = zeros(1, N);
elseif size(a, 2) == 1
    N = size(a, 1);
    b = zeros(N, 1);
else
    display('x is not a vector')
    b = [];
    return;
end

u = zeros(1, N);

for i = 1:1:N/2
    u(i) = a(2*i - 1);
    u(N-i+1) = a(2*i);
end

y = dft1D(u);
% calculation of dct
for k = 1:N
    b(k) = real(phi(k-1, N)*exp(-1i*pi*(k-1)/(2*N))*y(k));
end

% function phi
function r = phi( ksi, K )
if ksi == 0
    r = 1/sqrt(K);
else
    r = sqrt(2/K);
end
end
end
 
 