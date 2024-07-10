#ifndef FHT_H
#define FHT_H

//#define _USE_MATH_DEFINES

#include <vector>
#include <dev_array.h>

namespace RapiDHT {
	enum Directions {
		DIRECTION_X,
		DIRECTION_Y,
		DIRECTION_Z
	};
	enum Modes {
		CPU,
		GPU,
		RFFT
	};

	class HartleyTransform {
	public:
		HartleyTransform(int rows, int cols = 0, int depth = 0, Modes mode = Modes::CPU)
			: _rows(rows), _cols(cols), _depth(depth), _mode(mode) {
			if (_rows <= 0 || _cols < 0 || _depth < 0) {
				throw std::invalid_argument("Error (initialization): at least Rows must be positive. \
					Cols and Depth can be zero (by default) but not negative.");
			}
			else if (_cols == 0 && _depth > 0) {
				throw std::invalid_argument("Error (initialization): if cols is zero, depth must be zero.");
			}

			// Preparation to 1D transforms
			if (_mode == Modes::CPU || _mode == Modes::RFFT) {
				_bit_reversed_indices_x.resize(_rows);
				_bit_reversed_indices_y.resize(_cols);
				_bit_reversed_indices_z.resize(_depth);
				bitReverse(&_bit_reversed_indices_x);
				bitReverse(&_bit_reversed_indices_y);
				bitReverse(&_bit_reversed_indices_z);
			}
			if (_mode == Modes::GPU) {
				// Initialize Vandermonde matrice on the host
				initializeKernelHost(&_h_Vandermonde_Matrix_x, rows);
				//initializeKernelHost(h_A, rows);
				//initializeKernelHost(h_A, rows);


				// transfer CPU -> GPU
				_d_Vandermonde_Matrix_x.resize(_rows * _cols);
				_d_Vandermonde_Matrix_x.set(&_h_Vandermonde_Matrix_x[0], _rows * _cols);
			}
		}
		void ForwardTransform(double* data);
		void InverseTransform(double* data);

	private:
		/* ------------------------- ND Transforms ------------------------- */
		/**
		 * FDHT1D(double* vector) returns the Hartley transform
		 * of an 1D array using a fast Hartley transform algorithm.
		 */
		void FDHT1D(double* vector, const Directions direction = Directions::DIRECTION_X);

		/**
		 * FHT2D(double* image_ptr) returns the Hartley transform
		 * of an 2D array using a fast Hartley transform algorithm. The 2D transform
		 * is equivalent to computing the 1D transform along each dimension of image.
		 */
		void FDHT2D(double* image);

		/**
		* DHT1DCuda(double* h_x, double* h_A, const int length) returns the Hartley
		* transform of an 1D array using a matrix x vector multiplication.
		*/
		void DHT1DCuda(double* h_x, double* h_A, const int length);

		/**
		* DHT2DCuda(double* image) returns the Hartley
		* transform of an 1D array using a matrix x matrix multiplication.
		*/
		void DHT2DCuda(double* image);

		/**
		 * RealFFT1D(double* vector) returns the Fourier transform
		 * of an 1D array using a real Fourier transform algorithm.
		 */
		void RealFFT1D(double* vector, const Directions direction = Directions::DIRECTION_X);

		void series1D(double* image, const Directions direction);

		static void bitReverse(std::vector<size_t>* indices);
		static void initializeKernelHost(std::vector<double>* kernel, const int cols);
		static std::vector<double> DHT1D(const std::vector<double>& a, const std::vector<double>& kernel);
		template <typename T>
		static void transpose(std::vector<std::vector<T>>* image);
		static void transposeSimple(double* image, int rows, int cols);

		size_t* chooseRevercedIndices(int* length, const Directions direction);

		int _rows = 0;
		int _cols = 0;
		int _depth = 0;
		Modes _mode = Modes::CPU;
		std::vector<size_t> _bit_reversed_indices_x;
		std::vector<size_t> _bit_reversed_indices_y;
		std::vector<size_t> _bit_reversed_indices_z;
		std::vector<double> _h_Vandermonde_Matrix_x;
		dev_array<double> _d_Vandermonde_Matrix_x;
	};
}

#endif // !FHT_H