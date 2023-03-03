#define _USE_MATH_DEFINES

#include "time.h"
#include <algorithm>
#include <bitset>
#include <dev_array.h>
#include <kernel.h>
#include <iostream>
#include <math.h>
#include <vector>

// using DataType = unsigned __int8;
using DataType = float;

void initializeKernelHost(std::vector<DataType>& kernel, const int cols);
void HT2DCuda(const std::vector<DataType>& X, std::vector<DataType>& Y, const int cols);
