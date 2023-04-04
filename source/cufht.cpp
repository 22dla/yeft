#include <cufht.h>
#include "time.h"
#include <algorithm>
#include <dev_array.h>
#include <kernel.h>
#include <iostream>
#include <math.h>
#include <utilities.h>

void DHT1DCuda(double* h_x, double* h_A, const int length) {
	// Allocate memory on the device
	dev_array<double> d_A(length * length);	// matrix for one line
	dev_array<double> d_x(length);			// input vector
	dev_array<double> d_y(length);			// output vector

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

