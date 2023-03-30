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

	// Initialize the matrice on the host
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
void DHT2DCuda(double* h_X, const int rows, const int cols) {
	// Allocate memory on the host
	std::vector<double> h_A(rows * cols);

	// Allocate memory on the device
	dev_array<double> d_A(rows * cols); // matrix for one line
	dev_array<double> d_X(rows * cols); // one slice
	dev_array<double> d_Y(rows * cols); // one slice

	// Initialize matrices on the host
	initializeKernelHost(h_A, rows);
	// transfer CPU -> GPU
	d_A.set(&h_A[0], rows * cols);

	// transfer CPU -> GPU
	d_X.set(&h_X[0], rows * cols);
	matrixMultiplication(d_A.getData(), d_X.getData(), d_Y.getData(), cols);
	// transfer GPU -> CPU
	d_Y.get(&h_X[0], rows * cols);
	cudaDeviceSynchronize();
}
