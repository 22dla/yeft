#include <cufht.h>
#include "time.h"
#include <algorithm>
//#include <bitset>
#include <dev_array.h>
#include <kernel.h>
#include <iostream>
#include <math.h>

void initializeKernelHost(std::vector<float>& kernel, const int cols) {
	const float m_pi = std::acos(-1);

	// Initialize matrices on the host
	for (size_t k = 0; k < cols; ++k) {
		for (size_t j = 0; j < cols; ++j) {
			kernel[k * cols + j] = cosf(2 * m_pi * k * j / cols) + sinf(2 * m_pi * k * j / cols);
		}
	}
}

void DHT1DCuda(float* h_x, const int length) {
	// Allocate memory on the host
	std::vector<float> h_A(length * length);

	// Allocate memory on the device
	dev_array<float> d_A(length * length);	// matrix for one line
	dev_array<float> d_x(length);			// input vector
	dev_array<float> d_y(length);			// output vector

	// Initialize matrices on the host
	initializeKernelHost(h_A, length);
	// transfer CPU -> GPU
	d_A.set(&h_A[0], length * length);
	// transfer CPU -> GPU
	d_x.set(h_x, length * length);
	vectorMatrixMultiplication(d_A.getData(), d_x.getData(), d_y.getData(), length);
	// transfer GPU -> CPU
	d_y.get(h_x, length);
	cudaDeviceSynchronize();
}

// template <typename T>
void HT2DCuda(const std::vector<float>& X, std::vector<float>& Y, const int cols, const int image_num) {
	// Allocate memory on the host
	std::vector<float> h_A(cols * cols);

	// Allocate memory on the device
	dev_array<float> d_A(cols * cols); // matrix for one line
	dev_array<float> d_X(cols * cols); // one slice
	dev_array<float> d_Y(cols * cols); // one slice

	// Initialize matrices on the host
	initializeKernelHost(h_A, cols);
	// transfer CPU -> GPU
	d_A.set(&h_A[0], cols * cols);

	for (int i0 = 0; i0 < image_num; ++i0) {
		for (int direction = 0; direction < 2; ++direction) {
			// transfer CPU -> GPU
			d_X.set(&X[i0 * cols * cols], cols * cols);
			matrixMultiplication(d_A.getData(), d_X.getData(), d_Y.getData(), cols);
			// transfer GPU -> CPU
			d_Y.get(&Y[0], cols * cols);
			cudaDeviceSynchronize();
		}
	}

	cudaDeviceSynchronize();
}
