#ifndef FHT_H
#define FHT_H

#define _USE_MATH_DEFINES
// #define MPIDataType MPI_UNSIGNED_CHAR
// #define MPIDataType MPI_REAL
#define PARALLEL

#include <vector>
#include <stdexcept>
#include <functional>

// using DataType = unsigned __int8;
// using DataType = float;

namespace RapiDHT {
	enum Directions {
		DIRECTION_X,
		DIRECTION_Y,
		DIRECTION_Z
	};
	enum Modes {
		CPU,
		GPU
	};

	class HartleyTransform {
	public:
		HartleyTransform(int rows, int cols = 0, int depth = 0, Modes mode = Modes::CPU)
			: rows_(rows)
			, cols_(cols)
			, depth_(depth) {
			if (rows_ <= 0 || cols_ < 0 || depth_ < 0) {
				throw std::invalid_argument("Error (initialization): at least Rows must be positive. \
					Cols and Depth can be zero (by default) but not negative.");
			} else if (cols_ == 0 && depth_ > 0) {
				throw std::invalid_argument("Error (initialization): if cols is zero, depth must be zero.");
			}

			// Preparation to 1D transforms
			bit_reversed_indices_x_.resize(rows_);
			bit_reversed_indices_y_.resize(cols_);
			bit_reversed_indices_z_.resize(depth_);
			bit_reverse(&bit_reversed_indices_x_);
			bit_reverse(&bit_reversed_indices_y_);
			bit_reverse(&bit_reversed_indices_z_);

		}
		void ForwardTransform(float* data);
		void InverseTransform(float* data);

	private:
		/* ------------------------- ND Transforms ------------------------- */
		/**
		 * FDHT1D(std::vector<float>* vector_ptr) returns the Hartley
		 * transform of an 1D array using a fast Hartley transform algorithm.
		 */
		void FDHT1D(std::vector<float>* vector, const Directions direction = Directions::DIRECTION_X);

		/**
		 * FDHT1D(float* vector, const int length) returns the Hartley
		 * transform of an 1D array using a fast Hartley transform algorithm.
		 */
		void FDHT1D(float* vector, const Directions direction = Directions::DIRECTION_X);

		/**
		 * FHT2D(std::vector<std::vector<float>>* image_ptr) returns the Hartley
		 * transform of an 2D array using a fast Hartley transform algorithm. The 2D transform
		 * is equivalent to computing the 1D transform along each dimension of image.
		 */
		void FDHT2D(std::vector<std::vector<float>>* image_ptr);

		/**
		 * FHT2D(float* image_ptr, const int rows) returns the Hartley
		 * transform of an 2D array using a fast Hartley transform algorithm. The 2D transform
		 * is equivalent to computing the 1D transform along each dimension of image.
		 */
		void FDHT2D(float* image, const int rows, const int cols);


		void series1d(std::vector<std::vector<float>>* image, const Directions direction);
		void series1d(float* image, const Directions direction);

		static void bit_reverse(std::vector<size_t>* indices);
		static void initialize_kernel_host(std::vector<float>* kernel, const int cols);
		static std::vector<float> DHT1D(const std::vector<float>& a, const std::vector<float>& kernel);
		template <typename T>
		static void transpose(std::vector<std::vector<T>>* image);
		static void transpose_simple(float* image, int rows, int cols);

		int rows_ = 0;
		int cols_ = 0;
		int depth_ = 0;
		std::vector<size_t> bit_reversed_indices_x_;
		std::vector<size_t> bit_reversed_indices_y_;
		std::vector<size_t> bit_reversed_indices_z_;

	};
}

#endif // !FHT_H