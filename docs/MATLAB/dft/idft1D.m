function a = idft1D( b )
% b - input vector
% a - result of ifft(b)

if size(b, 1) == 1
    N = size(b, 2);
    a = zeros(1, N);
elseif size(b, 2) == 1
    N = size(b, 1);
    a = zeros(N, 1);
else
    display('x is not a vector')
    a = [];
    return;
end

for k = 1:N
    for j = 1:N
        a(k) = a(k) + b(j)*exp(2*pi*1i*(k-1)*(j-1)/N);
    end
end

a = a/N;

if abs(sum(imag(a))) < 1e-8
    a = real(a);
end

end
 
 