function a = idft2D( b )
% b - input matrix
% a - result of ifft2(b)

[N, M] = size(b);

a = zeros(N, M);

for i = 1:N
    a(i, :) = idft1D(b(i, :));
end

for i = 1:M
    a(:, i) = idft1D(a(:, i));
end

end
 
 