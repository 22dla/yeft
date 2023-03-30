#include <utilities.h>
#include <fstream>
#include <iomanip>
#include <iostream>

template <typename T>
void print_data_1d(const std::vector<T>& data) {

	for (size_t idx = 0; idx < data.size(); ++idx) {
		std::cout << std::fixed << std::setprecision(2) << data[idx] << "\t";
	}
	std::cout << std::endl;
}

template<typename T>
void print_data_1d(const T* data, int length) {

	for (size_t idx = 0; idx < length; ++idx) {
		std::cout << std::fixed << std::setprecision(2) << data[idx] << "\t";
	}
	std::cout << std::endl;
}

template<typename T>
void write_matrix_to_csv(const T* matrix, const size_t rows,
	const size_t cols, const std::string& file_path) {
	std::ofstream output_file(file_path);
	if (!output_file) {
		throw std::runtime_error("Failed to open file for writing");
	}

	// Write matrix elements to file
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			//output_file << std::fixed << std::setprecision(2) << matrix[i * cols + j];
			output_file << matrix[i * cols + j];

			if (j < cols - 1) {
				output_file << ";";
			}
		}
		output_file << "\n";
	}

	output_file.close();
}

template <typename T>
std::vector<std::vector<std::vector<T>>> make_data_3d_vec_vec_vec(
	int n, int m, int l) {
	const double kPi = std::acos(-1);
	std::vector<std::vector<std::vector<T>>> data(l);

	for (size_t j1 = 0; j1 < l; ++j1) {
		data[j1].resize(n);
		for (size_t j2 = 0; j2 < n; ++j2) {
			data[j1][j2].resize(m);
			for (size_t j3 = 0; j3 < m; ++j3) {
				data[j1][j2][j3] = static_cast<T>(n + std::cos(j1 / kPi) 
					- std::sin(std::cos(j2)) + std::tan(j3) + 2 + l) / m;
			}
		}
	}
	return data;
}

template <typename T>
std::vector<T> make_data_1d(int rows) {
	if (rows < 0) {
		throw std::invalid_argument("Error: rows must be non-negative");
	}

	const double kPi = std::acos(-1);
	std::vector<T> data(rows);

	for (size_t idx = 0; idx < rows; ++idx) {
		data[idx] = static_cast<T>(rows + std::cos(idx / kPi) -
			std::sin(std::cos(idx)) + std::tan(idx) + 2 + idx * idx) / rows;
		//data[idx] = idx; // for debug
	}
	return data;
}

template <typename T>
std::vector<T> make_data_3d(int rows, int cols, int depth) {
	if (rows < 0 || cols < 0 || depth < 0) {
		throw std::invalid_argument("Error: rows, cols, and depth must be non-negative");
	}

	const double kPi = 3.14159265358979323846f;
	std::vector<T> data(rows * cols * depth);

	for (size_t k = 0; k < depth; ++k) {
		for (size_t j = 0; j < cols; ++j) {
			for (size_t i = 0; i < rows; ++i) {
				size_t idx = k * rows * cols + j * rows + i;
				//data[idx] = static_cast<T>(rows + cosf(k / kPi) - sinf(cosf(j)) + tanf(i) + 2 + depth) / cols;
				data[idx] = idx; // for debug
			}
		}
	}
	return data;
}


void show_time(double startTime, double finishTime, std::string message) {
	std::cout << message + ":\t" << finishTime - startTime << " sec" << std::endl;
}

// explicit template instantiation for int and double
template void print_data_1d(const std::vector<double>& data);
template void print_data_1d(const std::vector<int>& data);
template void write_matrix_to_csv(const double* matrix, const size_t rows,
	const size_t cols, const std::string& file_path);
template void write_matrix_to_csv(const int* matrix, const size_t rows,
	const size_t cols, const std::string& file_path);
template std::vector<int> make_data_1d(int rows);
template std::vector<double> make_data_1d(int rows);
template std::vector<int> make_data_3d(int rows, int cols, int depth);
template std::vector<double> make_data_3d(int rows, int cols, int depth);
template std::vector<std::vector<std::vector<int>>> make_data_3d_vec_vec_vec(int n, int m, int l);
template std::vector<std::vector<std::vector<double>>> make_data_3d_vec_vec_vec(int n, int m, int l);