function b = dftND( a )
% a - input ND data 
% b - result of fftn(a)


sizes = size(a);
n = size(sizes, 2);

if n == 1
    b = dft1D(a);
elseif n == 2
    b = dft2D(a);
elseif n == 3
    b = dft3D(a);
else
    display('dimension of vector > 3')
    b = [];
end

end
 
 