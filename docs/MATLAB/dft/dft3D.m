function b = dft3D( a )
% a - input 3D data
% b - result of fft(a)

[N, M, L] = size(a);
b = zeros(N, M, L);

%x-axis and y-axis fft 
for z = 1:L
     b(:, :, z) = dft2D(a(:, :, z)); 
end

%z-axis fft 
for y = 1:N
    for x = 1:M
        q = b(y, x, :);
        q = permute(q, [3, 2, 1]);
        q = dft1D(q);
        b(y, x, :) = q;
    end
end

% H = zeros(N, M, L);
% for z = 1:L
%     for y = 1:N
%         for x = 1:M
%             for zz = 1:L
%                 for yy = 1:N
%                     for xx = 1:M
%                         H(y, x, z) = H(y, x, z) + a(yy, xx, zz)...
%                             *exp(-2*pi*1i*...
%                             (((x-1)*(xx-1)/M) + ((y-1)*(yy-1)/N) + ((z-1)*(zz-1)/L)));
%                     end
%                 end
%             end
%         end
%     end
% end
end
 
 