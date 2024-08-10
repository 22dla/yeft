#ifndef KERNEL_CUH_
#define KERNEL_CUH_

#include <stdint.h>

void matrixMultiplication(const double *A, const double *B, double *C, int N);
void vectorMatrixMultiplication(const double* A, const double* B, double* C, int N);
void matrixTranspose(double* A, int N);
void BracewellTransform2D(double* A, int N);

#endif
