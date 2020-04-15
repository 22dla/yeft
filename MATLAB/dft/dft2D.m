function b = dft2D( a )
% a - input matrix
% b - result of fft(a)

[N, M] = size(a);

b = zeros(N, M);

for i = 1:N
    b(i, :) = dft1D(a(i, :));
end

for i = 1:M
    b(:, i) = dft1D(b(:, i));
end

end
 
 