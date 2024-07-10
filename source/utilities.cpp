#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <utilities.h>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <filesystem>

// Функция для загрузки изображений и создания 3D данных
std::vector<std::vector<std::vector<uint8_t>>> loadImagesTo3DArray(const std::string& folderPath) {
	std::vector<cv::Mat> images;
	for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
		if (entry.is_regular_file() && (entry.path().extension() == ".jpg"
			|| entry.path().extension() == ".png"
			|| entry.path().extension() == ".bmp")) {
			cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);

			if (img.empty()) {
				std::cerr << "Error loading: " << entry.path().string() << std::endl;
				continue;
			}

			images.push_back(img);
		}
	}

	if (images.empty()) {
		std::cerr << "No images for loading." << std::endl;
		return {};
	}

	int numImages = images.size();
	int height = images[0].rows;
	int width = images[0].cols;

	std::vector<std::vector<std::vector<uint8_t>>> image3D(numImages, std::vector<std::vector<uint8_t>>(height, std::vector<uint8_t>(width)));
	for (int i = 0; i < numImages; ++i) {
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				image3D[i][y][x] = images[i].at<uint8_t>(y, x);
			}
		}
	}

	return image3D;
}

template<typename T>
void print_data_1d(const T* data, int length) {

	for (size_t idx = 0; idx < length; ++idx) {
		std::cout << std::fixed << std::setprecision(2) << data[idx] << "\t";
	}
	std::cout << std::endl;
}

template<typename T>
void print_data_2d(const T* data, int rows_, int cols_) {

	for (size_t i = 0; i < rows_; ++i) {
		for (size_t j = 0; j < cols_; ++j) {
			std::cout << std::fixed << std::setprecision(2) << data[i * cols_ + j] << " ";
		}
		std::cout << "\n";
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
std::vector<std::vector<std::vector<T>>> makeData3DArray(
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
std::vector<T> makeData(std::initializer_list<int> sizes) {
	int num_dims = sizes.size();
	std::vector<int> dim_sizes(sizes);
	for (int i = 0; i < num_dims; i++) {
		if (dim_sizes[i] < 0) {
			throw std::invalid_argument("Invalid size");
		}
	}
	std::vector<T> data(1);
	for (int i = 0; i < num_dims; i++) {
		data.resize(data.size() * dim_sizes[i]);
	}
	// fill massive with random values
	for (int idx = 0; idx < data.size(); ++idx) {
		data[idx] = static_cast<T>(dim_sizes[0] + std::cos(std::asin(0.1) / (idx + 1)) -
			std::sin(std::cos(idx / dim_sizes[0])) +
			std::tan(idx * dim_sizes[0]) + 2 + idx) / (dim_sizes[0] * dim_sizes[0]);
	}
	//std::iota(data.begin(), data.end(), 0);

	return data;
}

void show_time(double startTime, double finishTime, std::string message) {
	std::cout << message + ":\t" << finishTime - startTime << " sec" << std::endl;
}

// explicit template instantiation for int and double
template void print_data_1d(const int* data, int length);
template void print_data_1d(const double* data, int length);
template void print_data_2d(const int* data, int rows, int cols);
template void print_data_2d(const double* data, int rows, int cols);
template void write_matrix_to_csv(const double* matrix, const size_t rows,
	const size_t cols, const std::string& file_path);
template void write_matrix_to_csv(const int* matrix, const size_t rows,
	const size_t cols, const std::string& file_path);
template std::vector<int> makeData(std::initializer_list<int> sizes);
template std::vector<double> makeData(std::initializer_list<int> size);
template std::vector<std::vector<std::vector<int>>> makeData3DArray(int n, int m, int l);
template std::vector<std::vector<std::vector<double>>> makeData3DArray(int n, int m, int l);
