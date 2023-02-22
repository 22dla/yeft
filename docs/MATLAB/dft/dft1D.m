function b = dft1D( a )
% a - input vector
% b - result of fft(x)

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


for k = 1:N
    for j = 1:N
        b(k) = b(k) + a(j)*exp(-2*pi*1i*(k-1)*(j-1)/N);
    end
end

end
 
 