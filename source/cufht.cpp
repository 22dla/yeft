#include <cufht.h>
#include "time.h"
#include <algorithm>
//#include <bitset>
#include <dev_array.h>
#include <kernel.h>
#include <iostream>
#include <math.h>
#include <utilities.h>

void initializeKernelHost(std::vector<double>& kernel, const int cols) {
	const double m_pi = std::acos(-1);

	// Initialize matrices on the host
	for (size_t k = 0; k < cols; ++k) {
		for (size_t j = 0; j < cols; ++j) {
			kernel[k * cols + j] = std::cos(2 * m_pi * k * j / cols) + std::sin(2 * m_pi * k * j / cols);
		}
	}
}

void DHT1DCuda(double* h_x, const int length) {
	// Allocate memory on the host
	std::vector<double> h_A(length * length);

	// Allocate memory on the device
	dev_array<double> d_A(length * length);	// matrix for one line
	dev_array<double> d_x(length);			// input vector
	dev_array<double> d_y(length);			// output vector

	// Initialize matrices on the host
	initializeKernelHost(h_A, length);

	//write_matrix_to_csv(h_A.data(), length, length, "matrix.csv");

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
void HT2DCuda(const std::vector<double>& X, std::vector<double>& Y, const int cols, const int image_num) {
	// Allocate memory on the host
	std::vector<double> h_A(cols * cols);

	// Allocate memory on the device
	dev_array<double> d_A(cols * cols); // matrix for one line
	dev_array<double> d_X(cols * cols); // one slice
	dev_array<double> d_Y(cols * cols); // one slice

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
