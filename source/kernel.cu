#include "kernel.h"
#include "device_launch_parameters.h"
#include "dev_array.h"

__global__ void matrixMultiplicationKernel(double* A, double* B, double* C, int N) {

	int ROW = blockIdx.y*blockDim.y + threadIdx.y;
	int COL = blockIdx.x*blockDim.x + threadIdx.x;

	double tmpSum = 0;

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

__global__ void matrixVectorMultKernel(double* A, double* x, double* y, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		double sum = 0.0f;
		for (int j = 0; j < N; j++) {
			sum += A[i * N + j] * x[j];
		}
		y[i] = sum;
	}
}

__global__ void matrixTransposeKernel(double* A, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < N && j < N && i < j) {
		double tmp = A[i * N + j];
		A[i * N + j] = A[j * N + i];
		A[j * N + i] = tmp;
	}
}

void matrixMultiplication(double *A, double *B, double *C, const int N) {
	// declare the number of blocks per grid and the number of threads per block
	// use 1 to 512 threads per block
	dim3 threadsPerBlock(N, N);
	dim3 blocksPerGrid(1, 1);
	if (N*N > 512) {
		threadsPerBlock.x = 512;
		threadsPerBlock.y = 512;
		//blocksPerGrid.x = ceil(double(N) / double(threadsPerBlock.x));
		//blocksPerGrid.y = ceil(double(N) / double(threadsPerBlock.y));
		
		//require check
		blocksPerGrid.x = (N + threadsPerBlock.x - 1) / threadsPerBlock.x;
		blocksPerGrid.y = (N + threadsPerBlock.y - 1) / threadsPerBlock.y;
	}

	matrixMultiplicationKernel <<< blocksPerGrid, threadsPerBlock >>> (A, B, C, N);
}

void vectorMatrixMultiplication(double* A, double* x, double* y, const int N) {

	int threadsPerBlock, blocksPerGrid;

	threadsPerBlock = (N > 512) ? 512 : N;
	blocksPerGrid = (N > 512) ? (N + threadsPerBlock - 1) / threadsPerBlock : 1;

	matrixVectorMultKernel <<< blocksPerGrid, threadsPerBlock >>> (A, x, y, N);
}

void matrixTranspose(double* A, const int N) {
	// declare the number of blocks per grid and the number of threads per block
	// use 1 to 512 threads per block
	dim3 threadsPerBlock(N, N);
	dim3 blocksPerGrid(1, 1);
	if (N * N > 512) {
		threadsPerBlock.x = 512;
		threadsPerBlock.y = 512;
		//blocksPerGrid.x = ceil(double(N) / double(threadsPerBlock.x));
		//blocksPerGrid.y = ceil(double(N) / double(threadsPerBlock.y));

		//require check
		blocksPerGrid.x = (N + threadsPerBlock.x - 1) / threadsPerBlock.x;
		blocksPerGrid.y = (N + threadsPerBlock.y - 1) / threadsPerBlock.y;
	}

	matrixTransposeKernel <<< blocksPerGrid, threadsPerBlock >> > (A, N);
}
