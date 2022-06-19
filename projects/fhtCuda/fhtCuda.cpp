#include <iostream>
#include <algorithm>
#include <bitset>
#include <math.h>
#include <vector>
#include "kernel.h"
#include "dev_array.h"


//using DataType = unsigned __int8;
using DataType = float;

void initializeKernelHost(std::vector<DataType>& kernel, const int cols)
{
	const DataType m_pi = 3.14159265358979323846f;

	// Initialize matrices on the host
	for (size_t k = 0; k < cols; ++k) {
		for (size_t j = 0; j < cols; ++j){
			kernel[k*cols + j] = cosf(2 * m_pi*k*j / cols) + sinf(2 * m_pi*k*j / cols);
		}
	}
}

int main()
{
	// Define global 3D array sizes
	const int cols = pow(2, 9);
	int SIZE = cols * cols;

	// Allocate memory on the host
	std::vector<DataType> h_A(SIZE);
	std::vector<DataType> h_B(cols);
	std::vector<DataType> h_C(cols);

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
	dev_array<DataType> d_A(SIZE);
	dev_array<DataType> d_B(cols);
	dev_array<DataType> d_C(cols);

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

	printf("Time to generate:  %3.7f s \n", time / 1000.0);
	return 0;
}