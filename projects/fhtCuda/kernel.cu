#include "kernel.h"
#include "device_launch_parameters.h"
#include "dev_array.h"

//using DataType = unsigned __int8;

__global__ void matrixMultiplicationKernel(DataType* A, DataType* B, DataType* C, int N) {

	int ROW = blockIdx.y*blockDim.y + threadIdx.y;
	int COL = blockIdx.x*blockDim.x + threadIdx.x;

	DataType tmpSum = 0;

	if (ROW < N && COL < N) {
		// each thread computes one element of the block sub-matrix
		for (int i = 0; i < N; i++) {
			tmpSum += A[ROW * N + i] * B[i * N + COL];
		}
	}
	C[ROW * N + COL] = tmpSum;
}

void matrixMultiplication(DataType *A, DataType *B, DataType *C, int N) {

	// declare the number of blocks per grid and the number of threads per block
	// use 1 to 512 threads per block
	dim3 threadsPerBlock(N, N);
	dim3 blocksPerGrid(1, 1);
	if (N*N > 512) {
		threadsPerBlock.x = 512;
		threadsPerBlock.y = 512;
		blocksPerGrid.x = ceil(double(N) / double(threadsPerBlock.x));
		blocksPerGrid.y = ceil(double(N) / double(threadsPerBlock.y));
	}

	matrixMultiplicationKernel <<< blocksPerGrid, threadsPerBlock >>> (A, B, C, N);
}
