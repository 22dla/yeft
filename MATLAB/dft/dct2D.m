function b = dct2D( a )
% a - input matrix
% b - result of fft(a)

[N, M] = size(a);

b = zeros(N, M);

for i = 1:N
    b(i, :) = dct1D(a(i, :));
end

for i = 1:M
    b(:, i) = dct1D(b(:, i));
end

end
 
 