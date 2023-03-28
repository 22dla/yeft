#define _USE_MATH_DEFINES

#include <vector>

// using DataType = unsigned __int8;
// using DataType = float;

void initializeKernelHost(std::vector<float>& kernel, const int cols);
/**
* DHT1DCuda(float* vector, const int length) returns the Hartley
* transform of an 1D array using a matrix x vector multiplication.
*/
void DHT1DCuda(float* vector, const int length);

void HT2DCuda(const std::vector<float>& X, std::vector<float>& Y, const int cols, const int image_num);
