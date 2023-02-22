function a = idft3D( b )
% b - input 3D data
% a - result of ifft3(b)

[N, M, L] = size(b);
a = zeros(N, M, L);

%x-axis and y-axis ifft 
for z = 1:L
     a(:, :, z) = idft2D(b(:, :, z)); 
end

%z-axis ifft 
for y = 1:N
    for x = 1:M
        q = a(y, x, :);
        q = permute(q, [3, 2, 1]);
        q = idft1D(q);
        a(y, x, :) = q;
    end
end

end
 
 