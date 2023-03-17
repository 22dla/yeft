#define _USE_MATH_DEFINES
// #define MPIDataType MPI_UNSIGNED_CHAR
// #define MPIDataType MPI_REAL


#include "time.h"
#include <algorithm>
#include <assert.h>
#include <bitset>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <vector>

// using DataType = unsigned __int8;
using DataType = float;

void bitReverse(std::vector<size_t>* indices);
void fht1d(std::vector<DataType>* a);
void initializeKernelHost(std::vector<DataType>* kernel, const int cols);
std::vector<DataType> dht1d(const std::vector<DataType>& a, const std::vector<DataType>& kernel);
void showTime(double startTime, double finishTime, std::string message);

template <typename T>
void writeData(const std::vector<T>& vec, int mode = std::ios_base::out,
    const std::string& name = "vector", const std::string& path = "vector.csv");
template <typename T>
void FHT3D(T*** cube, const size_t cols);