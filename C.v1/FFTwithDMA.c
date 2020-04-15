#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <time.h>

#include <openacc.h>

typedef double complex cplx;

void _fft(cplx buf[], cplx out[], int n, int step)
{
    //#pragma acc data present(buf,out,n,step)
    if (step < n)
    {
        _fft(out, buf, n, step * 2);
        _fft(out + step, buf + step, n, step * 2);

        for (int i = 0; i < n; i += 2 * step)
        {
            cplx t = cexp(-I * 3.1415926535 * i / n) * out[i + step];
            buf[i / 2] = out[i] + t;
            buf[(i + n) / 2] = out[i] - t;
        }
    }
}

void fft(cplx buf[], int n)
{
    cplx temp[n];
    cplx out[n];
    #pragma acc kernels
    for (int i = 0; i < n; i++)
    {
        temp[i] = buf[i];
        out[i] = buf[i];
    }

    _fft(temp, out, n, 1);
}

int main()
{
    const int n = 2048;
    const int m = 2048;

    cplx data[n][m];
    cplx buf[m];

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {

            if (j < m / 2)
            {
                data[i][j] = 1;
            }
            else
            {
                data[i][j] = 0;
            }
            // printf("%g ", creal(data[i][j]));
        }
        // printf("\n");
    }

    clock_t begin = clock();

    //#pragma acc parallel loop copyin(data) copy(buf) create(i, j, k, m) 
    
    for (int k = 0; k < 10; ++k)
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                buf[j] = exp(data[i][j]+cos(i)*sin(j));
            }
            fft(buf, m);
        }
    }

    clock_t end = clock();

    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    printf("Time taken by FFT: %f \n", time_spent);

    return 0;
}