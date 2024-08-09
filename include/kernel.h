#ifndef KERNEL_CUH_
#define KERNEL_CUH_

#include <stdint.h>

void matrixMultiplication(double *A, double *B, double *C, const int N);
void vectorMatrixMultiplication(double* A, double* B, double* C, const int N);
void vectorImageMultiplication(uint8_t* A, uint8_t* B, uint8_t* C, int N);
void matrixTranspose(double* A, const int N);

#endif
