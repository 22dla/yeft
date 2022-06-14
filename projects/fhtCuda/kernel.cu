#define HASH_SIZE 128
#define _USE_MATH_DEFINES
#define MPIDataType MPI_REAL

#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "kernel.cuh"
#include <stdlib.h>
#include "device_launch_parameters.h"
#include <algorithm>
#include <bitset>
#include <vector>
#include "dev_array.h"

using namespace std;

//using DataType = unsigned __int8;
using DataType = float;

void bitReverse(int *indices, const int length)
{
	const int log2n = (int)log2f(length);
	// array to store binary number
	bool *binaryNum = new bool[length];

	indices[0] = 0;
	for (int j = 1; j < length; ++j)
	{
		// counter for binary array
		int count = 0;
		int base = j;
		while (base > 0)
		{
			// storing remainder in binary array
			binaryNum[count] = base % 2;
			base /= 2;
			count++;
		}
		for (int i = count; i < log2n; ++i)
		{
			binaryNum[i] = 0;
		}

		int dec_value = 0;
		base = 1;
		for (int i = log2n - 1; i >= 0; i--)
		{
			if (binaryNum[i] == 1)
			{
				dec_value += base;
			}
			base = base * 2;
		}

		indices[j] = dec_value;
	}

	delete[] binaryNum;
}

void bitReverse(std::vector<size_t> &indices)
{
	const int log2n = (int)log2f(indices.size());
	// array to store binary number
	std::vector<bool>binaryNum(indices.size());

	indices[0] = 0;
	for (size_t j = 1; j < indices.size(); ++j)
	{
		// counter for binary array
		size_t count = 0;
		int base = (int)j;
		while (base > 0)
		{
			// storing remainder in binary array
			binaryNum[count] = (bool)base % 2;
			base /= 2;
			count++;
		}
		for (int i = count; i < log2n; ++i)
			binaryNum[i] = false;

		int dec_value = 0;
		base = 1;
		for (int i = log2n - 1; i >= 0; i--)
		{
			if (binaryNum[i])
			{
				dec_value += base;
			}
			base = base * 2;
		}

		indices[j] = dec_value;
	}
}

void fht(std::vector<DataType>& a)
{
	// FHT for 3rd axis
	size_t M = a.size();
	const int log2 = (int)log2f(M);
	const DataType m_pi = 3.14159265358979323846f;

	// Indices for bit reversal operation
	std::vector<size_t> newIndeces(M);
	bitReverse(newIndeces);

	for (int i = 1; i < M / 2; ++i)
	{
		std::swap(a[i], a[newIndeces[i]]);
	}

	for (int s = 1; s <= log2; ++s)
	{
		int m = powf(2, s);
		int m2 = m / 2;
		int m4 = m / 4;

		for (size_t r = 0; r <= M - m; r = r + m)
		{
			for (size_t j = 1; j < m4; ++j)
			{
				int k = m2 - j;
				DataType u = a[r + m2 + j];
				DataType v = a[r + m2 + k];
				DataType c = cosf((DataType)j * m_pi / (DataType)m2);
				DataType s = sinf((DataType)j * m_pi / (DataType)m2);
				a[r + m2 + j] = u * c + v * s;
				a[r + m2 + k] = u * s - v * c;
			}
			for (size_t j = 0; j < m2; ++j)
			{
				DataType u = a[r + j];
				DataType v = a[r + j + m2];
				a[r + j] = u + v;
				a[r + j + m2] = u - v;
			}
		}
	}

}

void initializeKernelHost(std::vector<DataType>& kernel, const int cols)
{
	const DataType m_pi = 3.14159265358979323846f;

	// Initialize matrices on the host
	for (size_t k = 0; k < cols; ++k) {
		for (size_t j = 0; j < cols; ++j) {
			kernel[k*cols + j] = cosf(2 * m_pi*k*j / cols) + sinf(2 * m_pi*k*j / cols);
		}
	}
}

DataType* dht(DataType *a, const std::vector<DataType>& kernel, const int cols)
{
	DataType *result = new DataType[cols]();

	for (size_t i = 0; i < cols; i++)
		for (size_t j = 0; j < cols; j++)
			result[i] += (kernel[i*cols + j] * a[j]);

	return result;
}


__global__ void matrixMultiplicationKernel(float* A, float* B, float* C, int N) {

	int ROW = blockIdx.y*blockDim.y + threadIdx.y;
	int COL = blockIdx.x*blockDim.x + threadIdx.x;

	float tmpSum = 0;

	if (ROW < N && COL < N) {
		// each thread computes one element of the block sub-matrix
		for (int i = 0; i < N; i++) {
			tmpSum += A[ROW * N + i] * B[i * N + COL];
		}
	}
	C[ROW * N + COL] = tmpSum;
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

int main()
{
	// Define global 3D array sizes
	const int cols = pow(2, 9);
	int SIZE = cols * cols;

	// Allocate memory on the host
	vector<float> h_A(SIZE);
	vector<float> h_B(cols);
	vector<float> h_C(cols);

	// input data
	for (int j = 0; j < cols; ++j)
	{
		//float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		float r = 1.0f;

		h_B[j] = (cols + j + 1 + r) / cols;
	}

	// DFT
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Allocate memory on the device
	dev_array<float> d_A(SIZE);
	dev_array<float> d_B(cols);
	dev_array<float> d_C(cols);

	// Initialize matrices on the host
	initializeKernelHost(h_A, cols);
	d_A.set(&h_A[0], SIZE);
	d_B.set(&h_B[0], cols);

	for (int direction = 0; direction < 3; ++direction) {
		for (int i = 0; i < cols; ++i) {
			for (int j = 0; j < cols; ++j) {
				matrixMultiplication(d_A.getData(), d_B.getData(), d_C.getData(), cols);
			}
		}
		cudaDeviceSynchronize();
	}

	

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	printf("Time to generate:  %3.7f s \n", time/1000.0);
	return 0;
}
