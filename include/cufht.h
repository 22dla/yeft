#define _USE_MATH_DEFINES

#include <vector>

// using DataType = unsigned __int8;
// using DataType = double;

void initializeKernelHost(std::vector<double>& kernel, const int cols);
/**
* DHT1DCuda(double* vector, const int length) returns the Hartley
* transform of an 1D array using a matrix x vector multiplication.
*/
void DHT1DCuda(double* vector, const int length);

void DHT2DCuda(double* matrix, const int rows, const int cols);
