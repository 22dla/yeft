#define _USE_MATH_DEFINES

#include <iostream>
#include <algorithm>
#include <bitset>
#include <math.h>
#include <vector>
#include "time.h"
#include "kernel.h"
#include "dev_array.h"

//using DataType = unsigned __int8;
//using DataType = float;

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

//template <typename T>
void HT2DCuda(const std::vector<DataType>& X, std::vector<DataType>& Y, const int cols)
{
	// Allocate memory on the host
	std::vector<DataType> h_A(cols * cols);

	// Allocate memory on the device
	dev_array<DataType> d_A(cols * cols);	// maatrix for one line
	dev_array<DataType> d_X(cols * cols);	// one slice
	dev_array<DataType> d_Y(cols * cols);	// one slice

	// Initialize matrices on the host
	initializeKernelHost(h_A, cols);
	// transfer CPU -> GPU
	d_A.set(&h_A[0], cols * cols);

	for (int direction = 0; direction < 2; ++direction) {
		// transfer CPU -> GPU
		d_X.set(&X[0], cols*cols);
		matrixMultiplication(d_A.getData(), d_X.getData(), d_Y.getData(), cols);
		// transfer GPU -> CPU
		d_Y.get(&Y[0], cols*cols);
		cudaDeviceSynchronize();
	}

	cudaDeviceSynchronize();
}

int main()
{
	// Define global ND array sizes
	const int cols = pow(2, 13);

	std::vector<DataType> h_B(cols * cols);
	std::vector<DataType> h_C(cols * cols);

	// input data
	for (int j1 = 0; j1 < cols*cols; ++j1)
	{
		//float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		float r = 1.0f;
		h_B[j1] = (cols + j1 + 1 + r) / cols;
	}

	
	float time1;
	clock_t commonStart, commonStop;
	cudaEvent_t start, stop;
	commonStart = clock();

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// DHT
	HT2DCuda(h_B, h_C, cols);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time1, start, stop);
	commonStop = clock();

	printf("GPU Time:  \t%3.3f sec \n", time1 / 1000.0);
	double time_taken = double(commonStop - commonStart) / double(CLOCKS_PER_SEC);
	printf("Common time:  \t%3.3f sec \n", time_taken);
	return 0;
}