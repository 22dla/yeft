#include "kernel.h"
#include "device_launch_parameters.h"
#include "dev_array.h"

//using DataType = unsigned __int8;

__global__ void matrixMultiplicationKernel(float* A, float* B, float* C, int N) {

	int ROW = blockIdx.y*blockDim.y + threadIdx.y;
	int COL = blockIdx.x*blockDim.x + threadIdx.x;

	float tmpSum = 0;

	if (ROW < N && COL < N) {
		// each thread computes one element of the block sub-matrix
		for (int i = 0; i < N; i++) {
			//tmpSum += A[ROW * N + i] * B[i];			// for A * b = c (b, c - vectors)
			tmpSum += A[ROW * N + i] * B[i * N + COL];	// for A * B = C (b, c - matrices)
		}
	}
	//C[ROW] = tmpSum;									// for A * b = c (b, c - vectors)
	C[ROW * N + COL] = tmpSum;							// for A * B = C (b, c - matrices)
}

__global__ void matrixVectorMult(float* A, float* x, float* y, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		float sum = 0.0f;
		for (int j = 0; j < N; j++) {
			sum += A[i * N + j] * x[j];
		}
		y[i] = sum;
	}
}

void matrixMultiplication(float *A, float *B, float *C, int N) {

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

void vectorMatrixMultiplication(float* A, float* x, float* y, int N) {

	int threadsPerBlock, blocksPerGrid;

	threadsPerBlock = (N > 512) ? 512 : N;
	blocksPerGrid = (N > 512) ? 1 : (N + threadsPerBlock - 1) / threadsPerBlock;

	matrixVectorMult <<<blocksPerGrid, threadsPerBlock >>> (A, x, y, N);
}
