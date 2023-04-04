#ifndef KERNEL_CUH_
#define KERNEL_CUH_

void matrixMultiplication(double *A, double *B, double *C, const int N);
void vectorMatrixMultiplication(double* A, double* B, double* C, const int N);
void matrixTranspose(double* A, const int N);

#endif
