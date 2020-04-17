/*+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
*	To compile: 
*	pgcc -acc -ta=tesla -Minfo=accel FFT_OpenACC_iterative.c
*	pgcc -acc -ta=multicore -Minfo=accel FFT_OpenACC_iterative.c
*	pgcc -acc -ta=host -Minfo=accel FFT_OpenACC_iterative.c
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+*/
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <omp.h>

// #pragma STDC CX_LIMITED_RANGE on

typedef double complex cplx;
const double PI = 3.1415926536;

void fillInputMatrix(int nlines, int ncol, cplx a[][ncol])
{
	for (int i = 0; i < nlines; ++i)
	{
		for (int j = 0; j < ncol; ++j)
		{
			if (j < ncol / 2)
			{
				a[i][j] = 1; //exp(1 + cos(i) * sin(j));
			}
			else
			{
				a[i][j] = 0; //exp(cos(i) * sin(j));
			}
		}
	}
}
void fillOutputMatrix(int nlines, int ncol, cplx a[][ncol])
{
	for (int i = 0; i < nlines; ++i)
	{
		for (int j = 0; j < ncol; ++j)
		{
			a[i][j] = 0; //exp(1 + cos(i) * sin(j));
		}
	}
}

void showMatrix(int nlines, int ncol, cplx a[][ncol])
{
	printf("Matrix: \n");
	for (int i = 0; i < nlines; ++i)
	{
		for (int j = 0; j < ncol; ++j)
		{
			printf("(%.1f, %.1f) ", creal(a[i][j]), cimag(a[i][j]));
		}
		printf("\n");
	}
}

int main()
{
	float begin, time;
	const int n = 12 * 2048; //Count of lines
	const int m = 2048;		 //Length of vectors

	const int log2n = log2f(m);

	// Declaring input, output arrays
	cplx dataIn[n][m];
	cplx dataOut[n][m];

	fillInputMatrix(n, m, dataIn);
	//fillOutputMatrix(n, m, dataOut);
	// showMatrix(1, m, dataIn);
	// showMatrix(n, m, dataOut);

#pragma acc data copyin(n, m, dataIn) copyout(dataOut)
	{
		begin = omp_get_wtime();
#pragma acc parallels loop
		for (int index = 0; index < n; ++index)
		{
			// bit reversal of the given array
			for (int j = 0; j < m; ++j)
			{
				int rev = 0;
				int x = j;
				for (int i1 = 0; i1 < log2n; i1++)
				{
					rev <<= 1;
					rev |= (x & 1);
					x >>= 1;
				}
				dataOut[index][j] = dataIn[index][rev];
			}
			// FFT main function
			for (int s = 1; s <= log2n; ++s)
			{
				int m1 = 1 << s;  // 2 power s
				int m2 = m1 >> 1; // m2 = m1/2 -1
				cplx w = 1;
				cplx wm = cexp(I * (PI / m2));
				for (int j = 0; j < m2; ++j)
				{

					for (int k = j; k < m; k += m1)
					{
						// t = twiddle factor
						cplx t = w * dataOut[index][k + m2];
						cplx u = dataOut[index][k];

						// similar calculating y[k]
						dataOut[index][k] = u + t;

						// similar calculating y[k+n/2]
						dataOut[index][k + m2] = u - t;
					}
					w *= wm;
				}
			}
		}

		time = omp_get_wtime() - begin;
	}
	printf("Time taken by FFT: %f \n", time);

	//showMatrix(n, m, dataOut);
	return 0;
}