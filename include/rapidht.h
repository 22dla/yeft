#ifndef FHT_H
#define FHT_H

#define _USE_MATH_DEFINES
// #define MPIDataType MPI_UNSIGNED_CHAR
// #define MPIDataType MPI_REAL
#define PARALLEL

#include <vector>

// using DataType = unsigned __int8;
// using DataType = float;

void bit_reverse(std::vector<int>* indices);
void initialize_kernel_host(std::vector<float>* kernel, const int cols);
std::vector<float> DHT1D(const std::vector<float>& a, const std::vector<float>& kernel);
template <typename T>
void transpose(std::vector<std::vector<T>>* image);
void transpose_simple(float* image, int rows, int cols);
void series1d(std::vector<std::vector<float>>* image);
void series1d(float* image, const int rows, const int cols);

/* ------------------------- ND Transforms ------------------------- */

/**
 * FDHT1D(std::vector<float>* vector_ptr) returns the Hartley
 * transform of an 1D array using a fast Hartley transform algorithm.
 */
void FDHT1D(std::vector<float>* vector);

/**
 * FHT2D(std::vector<std::vector<float>>* image_ptr) returns the Hartley
 * transform of an 2D array using a fast Hartley transform algorithm. The 2D transform
 * is equivalent to computing the 1D transform along each dimension of image.
 */
void FDHT2D(std::vector<std::vector<float>>* image_ptr);

/**
 * FDHT1D(float* vector, const int length) returns the Hartley
 * transform of an 1D array using a fast Hartley transform algorithm.
 */
void FDHT1D(float* vector, const int length);

/**
 * FHT2D(float* image_ptr, const int rows) returns the Hartley
 * transform of an 2D array using a fast Hartley transform algorithm. The 2D transform
 * is equivalent to computing the 1D transform along each dimension of image.
 */
void FDHT2D(float* image, const int rows, const int cols);

#endif // !FHT_H