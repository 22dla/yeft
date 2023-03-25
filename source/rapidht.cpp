#include <algorithm>
#include <bitset>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <rapidht.h>
#include <utilities.h>

void bit_reverse(std::vector<int>* indices_ptr)
{
    std::vector<int>& indices = *indices_ptr;
    const int kLog2n = static_cast<int>(log2f(indices.size()));

    // array to store binary number
    std::vector<bool> binary_num(indices.size());

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
        for (int i = count; i < kLog2n; ++i)
            binary_num[i] = false;

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
}

void initialize_kernel_host(std::vector<float>* kernel, const int cols)
{
    const float kPi = 3.14159265358979323846f;
    if (kernel->size() != cols * cols) {
        kernel->resize(cols * cols);
    }

    // Initialize matrices on the host
    for (size_t k = 0; k < cols; ++k) {
        for (size_t j = 0; j < cols; ++j) {
            (*kernel)[k * cols + j] = cosf(2 * kPi * k * j / cols) + sinf(2 * kPi * k * j / cols);
        }
    }
}

// test function
std::vector<float> DHT1D(const std::vector<float>& a, const std::vector<float>& kernel)
{
    std::vector<float> result(a.size());

    for (size_t i = 0; i < a.size(); i++)
        for (size_t j = 0; j < a.size(); j++)
            result[i] += (kernel[i * a.size() + j] * a[j]);

    // RVO works
    return result;
}

template <typename T>
void transpose(std::vector<std::vector<T>>* matrix_ptr)
{
    std::vector<std::vector<T>>& matrix = *matrix_ptr;

    const size_t rows = matrix.size();
    const size_t cols = matrix[0].size();

#ifdef PARALLEL
#pragma omp parallel for
#endif
    for (int i = 0; i < rows; ++i) {
#ifdef PARALLEL
#pragma omp parallel for
#endif
        for (int j = i + 1; j < cols; ++j) {
            std::swap(matrix[i][j], matrix[j][i]);
        }
    }
}

void transpose_simple(float* matrix, const int rows, const int cols)
{
    if (matrix == nullptr) {
        throw std::invalid_argument("The pointer to matrix is null.");
    }

    if (rows == cols) {
#ifdef PARALLEL
#pragma omp parallel for
#endif
        // Square matrix
        for (int i = 0; i < rows; ++i) {
#ifdef PARALLEL
#pragma omp parallel for
#endif
            for (int j = i + 1; j < cols; ++j) {
                std::swap(matrix[i * cols + j], matrix[j * cols + i]);
            }
        }
    } else {
        // Non-square matrix
        std::vector<float> transposed(rows * cols);
#ifdef PARALLEL
#pragma omp parallel for
#endif
        for (int i = 0; i < rows; ++i) {
#ifdef PARALLEL
#pragma omp parallel for
#endif
            for (int j = 0; j < cols; ++j) {
                transposed[j * rows + i] = matrix[i * cols + j];
            }
        }
        std::memcpy(matrix, transposed.data(), sizeof(float) * rows * cols);
    }
}

void series1d(std::vector<std::vector<float>>* image_ptr)
{
    std::vector<std::vector<float>>& image = *image_ptr;
#ifdef PARALLEL
#pragma omp parallel for
#endif
    for (int i = 0; i < image.size(); ++i) {
        FDHT1D(&image[i]);
    }
}

void series1d(float* image_ptr, const int rows, const int cols)
{
    if (image_ptr == nullptr) {
        throw std::invalid_argument("The pointer to image is null.");
    }
    if (rows < 0 || cols < 0) {
        throw std::invalid_argument("Error: rows, and cols must be non-negative");
    }
#ifdef PARALLEL
#pragma omp parallel for
#endif
    for (int i = 0; i < rows; ++i) {
        FDHT1D(image_ptr + i * cols, cols);
    }
}

/**
 * FDHT1D(std::vector<float>* vector_ptr) returns the Hartley
 * transform of an 1D array using a fast Hartley transform algorithm.
 */
void FDHT1D(std::vector<float>* vector_ptr)
{
    auto& vec = *vector_ptr;
    // FHT for 1rd axis
    size_t M = vec.size();
    const int kLog2n = (int)log2f(M);
    const float kPi = 3.14159265358979323846f;

    // Indices for bit reversal operation
    std::vector<int> new_indeces(M);
    bit_reverse(&new_indeces);

    for (int i = 1; i < M / 2; ++i) {
        std::swap(vec[i], vec[new_indeces[i]]);
    }

    for (int s = 1; s <= kLog2n; ++s) {
        int m = powf(2, s);
        int m2 = m / 2;
        int m4 = m / 4;

        for (size_t r = 0; r <= M - m; r = r + m) {
            for (size_t j = 1; j < m4; ++j) {
                int k = m2 - j;
                float u = vec[r + m2 + j];
                float v = vec[r + m2 + k];
                float c = cosf(static_cast<float>(j) * kPi / m2);
                float s = sinf(static_cast<float>(j) * kPi / m2);
                vec[r + m2 + j] = u * c + v * s;
                vec[r + m2 + k] = u * s - v * c;
            }
            for (size_t j = 0; j < m2; ++j) {
                float u = vec[r + j];
                float v = vec[r + j + m2];
                vec[r + j] = u + v;
                vec[r + j + m2] = u - v;
            }
        }
    }
}

/**
 * FDHT1D(float* vec, const int length) returns the Hartley
 * transform of an 1D array using a fast Hartley transform algorithm.
 */
void FDHT1D(float* vec, const int length)
{
    if (vec == nullptr) {
        throw std::invalid_argument("The pointer to vector is null.");
    }
    if (length < 0) {
        throw std::invalid_argument("Error: length must be non-negative");
    }

    // FHT for 1rd axis
    const int kLog2n = (int)log2f(length);
    const float kPi = 3.14159265358979323846f;

    // Indices for bit reversal operation
    std::vector<int> new_indeces(length);
    bit_reverse(&new_indeces);

    for (int i = 1; i < length / 2; ++i) {
        std::swap(vec[i], vec[new_indeces[i]]);
    }

    for (int s = 1; s <= kLog2n; ++s) {
        int m = powf(2, s);
        int m2 = m / 2;
        int m4 = m / 4;

        for (size_t r = 0; r <= length - m; r = r + m) {
            for (size_t j = 1; j < m4; ++j) {
                int k = m2 - j;
                float u = vec[r + m2 + j];
                float v = vec[r + m2 + k];
                float c = cosf(static_cast<float>(j) * kPi / m2);
                float s = sinf(static_cast<float>(j) * kPi / m2);
                vec[r + m2 + j] = u * c + v * s;
                vec[r + m2 + k] = u * s - v * c;
            }
            for (size_t j = 0; j < m2; ++j) {
                float u = vec[r + j];
                float v = vec[r + j + m2];
                vec[r + j] = u + v;
                vec[r + j + m2] = u - v;
            }
        }
    }
}

/**
 * FHT2D(std::vector<std::vector<float>>* image_ptr) returns the Hartley
 * transform of an 2D array using a fast Hartley transform algorithm. The 2D transform
 * is equivalent to computing the 1D transform along each dimension of image.
 */
void FDHT2D(std::vector<std::vector<float>>* image_ptr)
{
    auto& image = *image_ptr;

    // 1D transforms along X dimension
    PROFILE(X, {
        series1d(&image);
    });

    PROFILE(TRANSPOSE, {
        transpose(&image);
    });

    // 1D transforms along Y dimension
    PROFILE(Y, {
        series1d(&image);
    });

    PROFILE(TRANSPOSE2, {
        transpose(&image);
    });
}

/**
 * FHT2D(float* image_ptr, const int rows) returns the Hartley
 * transform of an 2D array using a fast Hartley transform algorithm. The 2D transform
 * is equivalent to computing the 1D transform along each dimension of image.
 */
void FDHT2D(float* image_ptr, int rows, int cols)
{
    if (image_ptr == nullptr) {
        throw std::invalid_argument("The pointer to image is null.");
    }
    if (rows < 0 || cols < 0) {
        throw std::invalid_argument("Error: rows, and cols must be non-negative");
    }

    // write_matrix_to_csv(image_ptr, rows, cols, "matrix1.txt");

    // 1D transforms along X dimension
    PROFILE(X, {
        series1d(image_ptr, rows, cols);
    });

    PROFILE(Transpose1, {
        transpose_simple(image_ptr, rows, cols);
    });

    // 1D transforms along Y dimension
    PROFILE(Y, {
        series1d(image_ptr, cols, rows);
    });

    PROFILE(Transpose2, {
        transpose_simple(image_ptr, cols, rows);
    });

    // write_matrix_to_csv(image_ptr, rows, cols, "matrix2.txt");
}
