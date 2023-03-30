#ifndef UTILITIES_H
#define UTILITIES_H

#include <chrono>
#include <string>
#include <vector>

#define PROFILE(name, code)                                                                                  \
    auto start_##name = std::chrono::high_resolution_clock::now();                                           \
    code auto end_##name = std::chrono::high_resolution_clock::now();                                        \
    auto duration_##name = std::chrono::duration_cast<std::chrono::microseconds>(end_##name - start_##name); \
    std::cout << "Time taken by " #name ": " << duration_##name.count() << " microseconds" << std::endl;

template<typename T>
void print_data_1d(const std::vector<T>& data);
template<typename T>
void print_data_1d(const T* data, int length);
template<typename T>
void write_matrix_to_csv(const T* matrix, const size_t rows, 
    const size_t cols, const std::string& file_path);
template <typename T>
std::vector<std::vector<std::vector<T>>> make_data_3d_vec_vec_vec(
    int cols, int rows, int depth);
template <typename T>
std::vector<T> make_data_1d(int rows);
template <typename T>
std::vector<T> make_data_3d(int rows, int cols, int depth);

void show_time(double startTime, double finishTime, std::string message);

#endif // !UTILITIES_H