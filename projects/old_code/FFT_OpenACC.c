/*+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
*	To compile: pgcc -acc -ta=tesla -Minfo=accel FFT_OpenACC.c
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+*/
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <time.h>

typedef double complex cplx;

int main()
{
    // n x m - image resolution
    const int n = 2048;
    const int m = 2048;

    // Declaring input, output arrays
    cplx dataIn[n][m];
    cplx dataOut[n][m];

    // Fill in the input array
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            if (j < m / 2)
            {
                dataIn[i][j] = exp(1 + cos(i) * sin(j));
            }
            else
            {
                dataIn[i][j] = exp(cos(i) * sin(j));
            }
            dataOut[i][j] = 0;
            // printf("%g ", creal(data[i][j]));
        }
        // printf("\n");
    }

    // Start time of FFT execution
    clock_t begin = clock();

    #pragma acc declare copyin(dataIn, n, m) copyout(dataOut)
    for (int i = 0; i < n; ++i)
    {
        // Computation out = FFT(buf)
        #pragma acc parallel loop
        for (int i1 = 0; i1 < m; ++i1)
        {
            for (int j1 = 0; j1 < m; ++j1)
            {
                dataOut[i][i1] = dataOut[i][i1] + dataIn[i][j1] * cexp(-2 * 3.1415926535 * I * i1 * j1 / m);
            }
        }
    }

    // for (int i = 0; i < m; ++i)
    // {
    //     printf("(%g, %g) \n", creal(dataOut[i]), cimag(dataOut[i]));
    // }

    // End time of FFT execution
    clock_t end = clock();

    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    printf("Time taken by FFT: %f \n", time_spent);

    return 0;
}