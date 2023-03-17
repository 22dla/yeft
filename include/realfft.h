#define HASH_SIZE 128
#define _USE_MATH_DEFINES
// #define MPIDataType MPI_UNSIGNED_CHAR
#define MPIDataType MPI_REAL
// #define PARALLEL

#include "time.h"
#include <algorithm>
#include <assert.h>
#include <bitset>
#include <math.h>
#include <omp.h>
#include <vector>
#include <complex>

// using DataType = unsigned __int8;
using DataType = float;
typedef std::complex<DataType> Complex;

void showTime(double startTime, double finishTime, std::string message);
void fft1d(std::vector<Complex>* a);
void fft1dv2(std::vector<Complex>* x);