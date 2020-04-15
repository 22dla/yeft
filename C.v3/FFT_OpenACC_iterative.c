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

#pragma STDC CX_LIMITED_RANGE on

typedef double complex cplx;
const double PI = 3.1415926536;

int main()
{
	const int n = 2048;
	const int m = 2048;
	// const int l = 50;
	const int log2n = log2f(m);

	// Declaring input, output arrays
	cplx dataIn[n][m];
	cplx dataOut[n][m];

	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < m; ++j)
		{
			if (j < m / 2)
			{
				dataIn[i][j] = 1; //exp(1 + cos(i) * sin(j));
			}
			else
			{
				dataIn[i][j] = 0; //exp(cos(i) * sin(j));
			}
			dataOut[i][j] = 0;
			// printf("(%.1f, %.1f)", creal(dataIn[i][j]), cimag(dataIn[i][j]));
		}
		// printf("\n");
	}

	clock_t begin = clock();

	// FFT
#pragma acc declare copyin(dataIn, n, m, log2n) copyout(dataOut)
#pragma acc parallel loop
	// #pragma acc kernels
	// for (int count = 0; count < l; ++count)
	// {
	for (int idx = 0; idx < n; ++idx)
	{
		// bit reversal of the given array
		for (int j = 0; j < m; ++j)
		{
			int newN = 0;
			int x = j;
			for (int i1 = 0; i1 < log2n; i1++)
			{
				newN <<= 1;
				newN |= (x & 1);
				x >>= 1;
			}
			int rev = newN;
			dataOut[idx][j] = dataIn[idx][rev];
		}
		const cplx J = I;

		for (int s = 1; s <= log2n; ++s)
		{
			int m1 = 1 << s;  // 2 power s
			int m2 = m1 >> 1; // m2 = m/2 -1
			cplx w = 1;
			cplx wm = cexp(J * (PI / m2));
			for (int j = 0; j < m2; ++j)
			{
				for (int k = j; k < m; k += m1)
				{
					// t = twiddle factor
					cplx t = w * dataOut[idx][k + m2];
					cplx u = dataOut[idx][k];

					// similar calculating y[k]
					dataOut[idx][k] = u + t;

					// similar calculating y[k+n/2]
					dataOut[idx][k + m2] = u - t;
				}
				w *= wm;
			}
		}
	}
	// }

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Time taken by FFT: %f \n", time_spent);

	/*for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < m; ++j)
		{
			printf("(%.1f, %.1f)", creal(dataOut[i][j]), cimag(dataOut[i][j]));
		}
		printf("\n");
	}*/
	return 0;
}