function a = idftND( b )
% b - input ND data 
% a - result of ifftn(b)

sizes = size(b);
n = size(sizes, 2);

if n == 1
    a = idft1D(b);
elseif n == 2
    a = idft2D(b);
elseif n == 3
    a = idft3D(b);
else
    display('dimension of vector > 3')
    a = [];
end

end
 
 