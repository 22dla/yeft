#ifndef UTILITIES_H
#define UTILITIES_H

#include <chrono>
#include <string>
#include <vector>
#include <iostream>

#ifdef DEBUG
#define PROFILE_FUNCTION() Profiler __profiler(__FUNCTION__)
#else
#define PROFILE_FUNCTION()
#endif

class Profiler {
public:
	Profiler(const std::string& functionName) : 
		m_functionName(functionName), m_startTime(std::chrono::high_resolution_clock::now()) {}
	~Profiler() {
		auto endTime = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - m_startTime).count();
		std::cout << m_functionName << " took " << duration << " microseconds" << std::endl;
	}
private:
	std::string m_functionName;
	std::chrono::high_resolution_clock::time_point m_startTime;
};

std::vector<std::vector<std::vector<uint8_t>>> loadImagesTo3DArray(const std::string& folderPath);

template<typename T>
void printData1D(const T* data, int length);

template<typename T>
void printData2D(const T* data, int rows, int cols);

template<typename T>
void writeMatrixToCSV(const T* matrix, const size_t rows,
	const size_t cols, const std::string& file_path);

template <typename T>
std::vector<std::vector<std::vector<T>>> makeData3DArray(
	int cols, int rows, int depth);

template <typename T>
std::vector<T> makeData(std::initializer_list<int> sizes);

void showTime(double startTime, double finishTime, std::string message);

#endif // !UTILITIES_H