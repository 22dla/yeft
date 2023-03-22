#ifndef UTILITIES_H
#define UTILITIES_H

#include <iostream>
#include <string>
#include <vector>

void show_time(double startTime, double finishTime, std::string message);

template <typename T>
void write_data(const std::vector<T>& vec, int mode = std::ios_base::out,
    const std::string& name = "vector", const std::string& path = "vector.csv");

template <typename T>
std::vector<std::vector<std::vector<T>>> make_data_3d(int cols, int rows, int layers);

#endif // !UTILITIES_H