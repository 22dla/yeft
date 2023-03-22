#ifndef FHT_H
#define FHT_H

#define _USE_MATH_DEFINES
// #define MPIDataType MPI_UNSIGNED_CHAR
// #define MPIDataType MPI_REAL

#include "time.h"
#include <algorithm>
#include <assert.h>
#include <bitset>
#include <math.h>
#include <omp.h>
#include <vector>

// using DataType = unsigned __int8;
// using DataType = float;

void bitReverse(std::vector<size_t>* indices);
void initializeKernelHost(std::vector<float>* kernel, const int cols);
std::vector<float> dht1d(const std::vector<float>& a, const std::vector<float>& kernel);
void transpose(std::vector<std::vector<float>>* image);

void FDHT1D(std::vector<float>* vector);
void FDHT2D(std::vector<std::vector<float>>* image);
void FDHT3D(float*** cube, const size_t cols);

struct Image;
#endif // !FHT_H