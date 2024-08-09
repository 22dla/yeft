#include <omp.h>
#include <rapidht.h>
#include <utilities.h>
#include <complex>
#include <array>
#include <kernel.h>

namespace RapiDHT {
	HartleyTransform::HartleyTransform(size_t cols, size_t rows, size_t depth, Modes mode)
		: _mode(mode) {
		if (rows == 0 && depth > 0) {
			throw std::invalid_argument("Error (initialization): if cols is zero, depth must be zero.");
		}

		// Preparation to 1D transforms
		if (_mode == Modes::CPU || _mode == Modes::RFFT) {
			_bit_reversed_indices_x = bitReverse(cols);
			if (rows > 0) {
				_bit_reversed_indices_y = bitReverse(rows);
			}
			if (depth > 0) {
				_bit_reversed_indices_z = bitReverse(depth);
			}
		}
		if (_mode == Modes::GPU) {
			// Initialize Vandermonde matrice on the host
			initializeKernelHost(&_h_hartley_matrix_x, cols);
			//initializeKernelHost(h_A, cols);
			//initializeKernelHost(h_A, cols);

			// transfer CPU -> GPU
			_d_hartley_matrix_x.resize(rows * cols);
			_d_hartley_matrix_x.set(&_h_hartley_matrix_x[0], rows * cols);
		}
	}

	void HartleyTransform::ForwardTransform(std::vector<double>& data) {

		switch (_mode) {
		case RapiDHT::CPU:
			if (rows() == 0 && depth() == 0) {
				FDHT1D(data.begin(), data.end(), _bit_reversed_indices_x);
			}
			else if (depth() == 0) {
				FDHT2D(data, { _bit_reversed_indices_x, _bit_reversed_indices_y });
			}
			else {
				FDHT3D(data, { _bit_reversed_indices_x, _bit_reversed_indices_y, _bit_reversed_indices_z });
			}
			break;

		case RapiDHT::GPU:
			if (rows() == 0 && depth() == 0) {
				DHT1DCuda(data.data(), _h_hartley_matrix_x.data(), cols());
			}
			else if (depth() == 0) {
				DHT2DCuda(data.data());
			}
			break;

		case RapiDHT::RFFT:
			if (rows() == 0 && depth() == 0) {
				RealFFT1D(data, _bit_reversed_indices_x);
			}
			else if (depth() == 0) {
				RealFFT2D(data, { _bit_reversed_indices_x, _bit_reversed_indices_y });
			}
			break;
		default:
			break;
		}
	}

	void HartleyTransform::InverseTransform(std::vector<double>& data) {
		ForwardTransform(data);

		double denominator = 0;
		if (cols() == 0 && depth() == 0) {	// 1D
			denominator = 1.0f / rows();
		}
		else if (depth() == 0) {			// 2D
			denominator = 1.0f / (rows() * cols());
		}
		else {							// 3D
			denominator = 1.0f / (rows() * cols() * depth());
		}
		for (auto& item : data) {
			item *= denominator;
		}
	}

	std::vector<size_t> HartleyTransform::bitReverse(size_t length) {
		if (length == 0) {
			return {};
		}
		const int kLog2n = static_cast<int>(log2f(static_cast<float>(length)));

		// result vector of indices
		std::vector<size_t> indices(length);

		// array to store binary number
		std::vector<bool> binary_num(length);

		indices[0] = 0;
		for (int j = 1; j < indices.size(); ++j) {
			// counter for binary array
			size_t count = 0;
			int base = j;
			while (base > 0) {
				// storing remainder in binary array
				binary_num[count] = static_cast<bool>(base % 2);
				base /= 2;
				++count;
			}
			for (size_t i = count; i < kLog2n; ++i) {
				binary_num[i] = false;
			}

			int dec_value = 0;
			base = 1;
			for (int i = kLog2n - 1; i >= 0; --i) {
				if (binary_num[i]) {
					dec_value += base;
				}
				base *= 2;
			}
			indices[j] = dec_value;
		}
		return indices;
	}

	void HartleyTransform::initializeKernelHost(std::vector<double>* kernel, const int cols) {
		if (kernel == nullptr) {
			throw std::invalid_argument("Error: kernell==nullptr (initializeKernelHost)");
		}
		auto& ker = *kernel;
		ker.resize(cols * cols);
		const double m_pi = std::acos(-1);

		// Initialize the matrice on the host
		for (size_t k = 0; k < cols; ++k) {
			for (size_t j = 0; j < cols; ++j) {
				ker[k * cols + j] = std::cos(2 * m_pi * k * j / cols) + std::sin(2 * m_pi * k * j / cols);
			}
		}
	}

	// formula function
	std::vector<double> HartleyTransform::DHT1D(
		const std::vector<double>& a, const std::vector<double>& kernel) {
		std::vector<double> result(a.size());

		for (size_t i = 0; i < a.size(); i++)
			for (size_t j = 0; j < a.size(); j++)
				result[i] += (kernel[i * a.size() + j] * a[j]);

		// RVO works
		return result;
	}

	template <typename T>
	void HartleyTransform::transpose(std::vector<std::vector<T>>* matrix_ptr) {
		std::vector<std::vector<T>>& matrix = *matrix_ptr;

		const size_t rows = matrix.size();
		const size_t cols = matrix[0].size();

	#pragma omp parallel for
		for (int i = 0; i < rows; ++i) {
		#pragma omp parallel for
			for (int j = i + 1; j < cols; ++j) {
				std::swap(matrix[i][j], matrix[j][i]);
			}
		}
	}

	void HartleyTransform::transpose(std::vector<double>& matrix, int cols, int rows) {
		PROFILE_FUNCTION();
		if (cols == rows) {// Для квадратных матриц память не выделяем
		#pragma omp parallel for
			for (int i = 0; i < cols; ++i) {
			#pragma omp parallel for
				for (int j = i + 1; j < cols; ++j) {
					int index1 = i * cols + j;
					int index2 = j * cols + i;
					std::swap(matrix[index1], matrix[index2]);
				}
			}
		}
		else {// Неквадратная матрица: создаем новый вектор для транспонированной матрицы
			std::vector<double> transposed(cols * rows);
		#pragma omp parallel for
			for (int i = 0; i < rows; ++i) {
			#pragma omp parallel for
				for (int j = 0; j < cols; ++j) {
					transposed[j * rows + i] = matrix[i * cols + j];
				}
			}
			matrix = std::move(transposed);
		}
	}

	std::vector<double> HartleyTransform::transpose3D(
		const std::vector<double>& input,
		size_t rows, size_t cols, size_t depth,
		const std::array<size_t, 3>& old_indices,
		const std::array<size_t, 3>& new_indices) {
		// Размеры в старом и новом порядке
		std::array<size_t, 3> old_dims = { rows, cols, depth };
		std::array<size_t, 3> new_dims = { old_dims[old_indices[0]], old_dims[old_indices[1]], old_dims[old_indices[2]] };

		std::vector<double> output(new_dims[0] * new_dims[1] * new_dims[2]);

		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < cols; ++j) {
				for (size_t k = 0; k < depth; ++k) {
					// Индексы в исходном массиве
					std::array<size_t, 3> old_idx = { i, j, k };
					size_t index_in = old_idx[old_indices[0]] * old_dims[1] * old_dims[2]
						+ old_idx[old_indices[1]] * old_dims[2]
						+ old_idx[old_indices[2]];

					// Индексы в новом массиве
					std::array<size_t, 3> new_idx = { old_idx[old_indices[new_indices[0]]],
													 old_idx[old_indices[new_indices[1]]],
													 old_idx[old_indices[new_indices[2]]] };
					size_t index_out = new_idx[0] * new_dims[1] * new_dims[2]
						+ new_idx[1] * new_dims[2]
						+ new_idx[2];

					output[index_out] = input[index_in];
				}
			}
		}

		return output;
	}

	template <typename Iter>
	void HartleyTransform::FDHT1D(Iter first, Iter last, const std::vector<size_t>& bit_reversed_indices) {
		if (first == last) {
			return;
		}

		auto length = std::distance(first, last);
		if (length != bit_reversed_indices.size()) {
			throw std::invalid_argument("Error: size of vec must be equal size of bit_reversed_indices");
		}
		if (length < 0) {
			std::cout << "Error: length must be non-negative." << std::endl;
			throw std::invalid_argument("Error: length must be non-negative.");
		}

		// Check that length is power of 2
		if (std::ceil(std::log2(length)) != std::floor(std::log2(length))) {
			std::cout << "Error: length must be a power of two." << std::endl;
			throw std::invalid_argument("Error: length must be a power of two.");
		}

		std::vector<typename std::iterator_traits<Iter>::value_type> vec(first, last);

		for (int i = 1; i < length; ++i) {
			size_t j = bit_reversed_indices[i];
			if (j > i) {
				std::swap(vec[i], vec[j]);
			}
		}

		// FHT for 1rd axis
		const int kLog2n = static_cast<int>(log2f(static_cast<float>(length)));
		const double kPi = std::acos(-1);

		// Main cicle
		for (int s = 1; s <= kLog2n; ++s) {
			int m = static_cast<int>(powf(2, s));
			int m2 = m / 2;
			int m4 = m / 4;

			for (size_t r = 0; r <= length - m; r = r + m) {
				for (size_t j = 1; j < m4; ++j) {
					int k = m2 - j;
					double u = vec[r + m2 + j];
					double v = vec[r + m2 + k];
					double c = std::cos(static_cast<double>(j) * kPi / m2);
					double s = std::sin(static_cast<double>(j) * kPi / m2);
					vec[r + m2 + j] = u * c + v * s;
					vec[r + m2 + k] = u * s - v * c;
				}
				for (size_t j = 0; j < m2; ++j) {
					double u = vec[r + j];
					double v = vec[r + j + m2];
					vec[r + j] = u + v;
					vec[r + j + m2] = u - v;
				}
			}
		}

		std::copy(vec.begin(), vec.end(), first);
	}

	void HartleyTransform::BracewellTransform2D(std::vector<double>& image, size_t cols, size_t rows) {
		PROFILE_FUNCTION();
		std::vector<double> H(rows * cols, 0.0);
	#pragma omp parallel for
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				double A = image[i * cols + j];
				double B = (i > 0 && j > 0) ? image[i * cols + (cols - j)] : A;
				double C = (i > 0 && j > 0) ? image[(rows - i) * cols + j] : A;
				double D = (i > 0 && j > 0) ? image[(rows - i) * cols + (cols - j)] : A;
				H[i * cols + j] = (A + B + C - D) / 2.0;
			}
		}

		image = std::move(H);
	}

	void HartleyTransform::FDHT2D(std::vector<double>& image, const std::vector<std::vector<size_t>>& bit_reversed_indices) {
		PROFILE_FUNCTION();
		// writeMatrixToCSV(image_ptr, rows, cols, "matrix1.txt");

		if (bit_reversed_indices.size() != 2) {
			throw std::invalid_argument("Error: bit_reversed_indices.size() must be 2");
		}

		auto cols = bit_reversed_indices[0].size();
		auto rows = bit_reversed_indices[1].size();

		if (image.size() != rows * cols) {
			throw std::invalid_argument("Error: invalid sizes");
		}

		// 1D transforms along X dimension
	#pragma omp parallel for
		for (int i = 0; i < rows; ++i) {
			// Вычисление начала и конца строки
			auto start = image.begin() + i * cols;
			auto end = start + cols;

			// Вызов FDHT1D для одной строки
			FDHT1D(start, end, bit_reversed_indices[0]);
		}

		transpose(image, cols, rows);

		// 1D transforms along Y dimension
	#pragma omp parallel for
		for (int i = 0; i < cols; ++i) {
			// Вычисление начала и конца строки
			auto start = image.begin() + i * rows;
			auto end = start + rows;

			// Вызов FDHT1D для одной строки
			FDHT1D(start, end, bit_reversed_indices[1]);
		}

		transpose(image, rows, cols);
		BracewellTransform2D(image, cols, rows);

		// writeMatrixToCSV(image_ptr, rows, cols, "matrix2.txt");
	}

	void HartleyTransform::FDHT3D(std::vector<double>& cube, const std::vector<std::vector<size_t>>& bit_reversed_indices) {
		//if (image_ptr == nullptr) {
		//	std::cout << "The pointer to image is null." << std::endl;
		//	throw std::invalid_argument("The pointer to image is null.");
		//}
		//if (rows() < 0 || cols() < 0) {
		//	std::cout << "Error: rows, and cols must be non-negative." << std::endl;
		//	throw std::invalid_argument("Error: rows, and cols must be non-negative.");
		//}

		//// writeMatrixToCSV(image_ptr, rows, cols, "matrix1.txt");

		//// 1D transforms along X dimension
		//series1D(image_ptr, DIRECTION_X);

		//transposeSimple(image_ptr, rows(), cols());

		//// 1D transforms along Y dimension
		//series1D(image_ptr, DIRECTION_Y);

		//transposeSimple(image_ptr, cols(), rows());

		//// 1D transforms along Y dimension
		//series1D(image_ptr, DIRECTION_Z);

		//transposeSimple(image_ptr, cols(), rows());

		// writeMatrixToCSV(image_ptr, rows, cols, "matrix2.txt");
	}

	// test functions
	void HartleyTransform::RealFFT1D(std::vector<double>& vec, const std::vector<size_t>& bit_reversed_indices) {

		// Indices for bit reversal operation
		// and length of vector depending of direction
		auto length = bit_reversed_indices.size();

		// Check that length is power of 2
		if (std::ceil(std::log2(length)) != std::floor(std::log2(length))) {
			std::cout << "Error: length must be a power of two." << std::endl;
			throw std::invalid_argument("Error: length must be a power of two.");
		}

		// RealFFT
		std::vector<std::complex<double>> x(length);
		for (int i = 0; i < length; i++) {
			x[i] = std::complex<double>(vec[i], 0);
		}
		unsigned int k = length;
		unsigned int n;
		const double kPi = std::acos(-1);
		double thetaT = kPi / length;
		std::complex<double> phiT = std::complex<double>(cos(thetaT), -sin(thetaT)), T;
		while (k > 1) {
			n = k;
			k >>= 1;
			phiT = phiT * phiT;
			T = 1.0L;
			for (unsigned int l = 0; l < k; l++) {
				for (unsigned int a = l; a < length; a += n) {
					unsigned int b = a + k;
					std::complex<double> t = x[a] - x[b];
					x[a] += x[b];
					x[b] = t * T;
				}
				T *= phiT;
			}
		}
		// Decimate
		unsigned int m = (unsigned int)log2(length);
		for (unsigned int a = 0; a < length; a++) {
			unsigned int b = a;
			// Reverse bits
			b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
			b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
			b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
			b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
			b = ((b >> 16) | (b << 16)) >> (32 - m);
			if (b > a) {
				std::complex<double> t = x[a];
				x[a] = x[b];
				x[b] = t;
			}
		}

		for (int i = 0; i < length; i++) {
			vec[i] = x[i].real();
		}
	}

	void HartleyTransform::RealFFT2D(std::vector<double>& image, const std::vector<std::vector<size_t>>& bit_reversed_indices) {
		return;
	}

	void HartleyTransform::DHT1DCuda(double* h_x, double* h_A, int length) {
		// Allocate memory on the device
		dev_array<double> d_A(length * length);	// matrix for one line
		dev_array<double> d_x(length);			// input vector
		dev_array<double> d_y(length);			// output vector

		//writeMatrixToCSV(h_A.data(), length, length, "matrix.csv");

		// transfer CPU -> GPU
		d_A.set(&h_A[0], length * length);
		// transfer CPU -> GPU
		d_x.set(h_x, length * length);
		vectorMatrixMultiplication(d_A.getData(), d_x.getData(), d_y.getData(), length);
		// transfer GPU -> CPU
		d_y.get(h_x, length);
		cudaDeviceSynchronize();
	}

	void HartleyTransform::DHT2DCuda(double* h_X) {
		// Allocate memory on the device
		dev_array<double> d_X(rows() * cols()); // one slice
		dev_array<double> d_Y(rows() * cols()); // one slice

		// transfer CPU -> GPU
		d_X.set(&h_X[0], rows() * cols());
		matrixMultiplication(_d_hartley_matrix_x.getData(), d_X.getData(), d_Y.getData(), cols());
		matrixTranspose(d_Y.getData(), cols());
		matrixMultiplication(_d_hartley_matrix_x.getData(), d_Y.getData(), d_X.getData(), cols());
		matrixTranspose(d_X.getData(), cols());

		// transfer GPU -> CPU
		d_X.get(&h_X[0], rows() * cols());
		cudaDeviceSynchronize();
	}
}