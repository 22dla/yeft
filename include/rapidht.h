#ifndef FHT_H
#define FHT_H

//#define _USE_MATH_DEFINES

#include <vector>
#include <dev_array.h>

namespace RapiDHT {
	enum Modes {
		CPU,
		GPU,
		RFFT
	};

	class HartleyTransform {
	public:
		HartleyTransform(size_t rows, size_t cols = 0, size_t depth = 0, Modes mode = Modes::CPU);

		void ForwardTransform(std::vector<double>& data);
		void InverseTransform(std::vector<double>& data);

		size_t rows() {
			return _bit_reversed_indices_x.size();
		}
		size_t cols() {
			return _bit_reversed_indices_y.size();
		}
		size_t depth() {
			return _bit_reversed_indices_z.size();
		}

	private:
		/* ------------------------- ND Transforms ------------------------- */
		/**
		 * FDHT1D(double* vector) returns the Hartley transform
		 * of an 1D array using a fast Hartley transform algorithm.
		 */
		template <typename Iter>
		static void FDHT1D(Iter begin, Iter end, const std::vector<size_t>& bit_reversed_indices);

		/**
		 * FHT2D(double* image_ptr) returns the Hartley transform
		 * of an 2D array using a fast Hartley transform algorithm. The 2D transform
		 * is equivalent to computing the 1D transform along each dimension of image.
		 */
		static void FDHT2D(std::vector<double>& image, const std::vector<std::vector<size_t>>& bit_reversed_indices);

		/**
		 * FHT2D(double* image_ptr) returns the Hartley transform
		 * of an 3D array using a fast Hartley transform algorithm. The 3D transform
		 * is equivalent to computing the 1D transform along each dimension of image.
		 */
		static void FDHT3D(std::vector<double>& cube, const std::vector<std::vector<size_t>>& bit_reversed_indices);

		/**
		* DHT1DCuda(double* h_x, double* h_A, const int length) returns the Hartley
		* transform of an 1D array using a matrix x vector multiplication.
		*/
		void DHT1DCuda(double* h_x, double* h_A, int length);

		/**
		* DHT2DCuda(double* image) returns the Hartley
		* transform of an 1D array using a matrix x matrix multiplication.
		*/
		void DHT2DCuda(double* image);

		/**
		 * RealFFT1D(double* vector) returns the Fourier transform
		 * of an 1D array using a real Fourier transform algorithm.
		 */
		static void RealFFT1D(std::vector<double>& vector, const std::vector<size_t>& bit_reversed_indices);
		static void RealFFT2D(std::vector<double>& image, const std::vector<std::vector<size_t>>& bit_reversed_indices);

		static std::vector<size_t> bitReverse(size_t length);
		static void initializeKernelHost(std::vector<double>* kernel, const int cols);
		static std::vector<double> DHT1D(const std::vector<double>& a, const std::vector<double>& kernel);
		template <typename T>
		static void transpose(std::vector<std::vector<T>>* image);
		static void transpose(std::vector<double>& matrix, int cols, int rows);
		static void transposeSimple(double* image, int rows, int cols);
		static std::vector<double> transpose3D(
			const std::vector<double>& input,
			size_t rows, size_t cols, size_t depth,
			const std::array<size_t, 3>& old_indices,
			const std::array<size_t, 3>& new_indices);

		Modes _mode = Modes::CPU;
		std::vector<size_t> _bit_reversed_indices_x;
		std::vector<size_t> _bit_reversed_indices_y;
		std::vector<size_t> _bit_reversed_indices_z;
		
		// TODO исправить название
		std::vector<double> _h_hartley_matrix_x;
		dev_array<double> _d_hartley_matrix_x;
	};
}

#endif // !FHT_H