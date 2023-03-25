#include <rapidht.h>
#include <fstream>
#include <iostream>

class Image {
public:
    Image(const std::vector<std::vector<float>>& other)
    { // copy constructor
        data = other;
    }

    int rows() const
    {
        return data[0].size();
    }
    int cols() const
    {
        return data.size();
    }
    float operator()(int x, int y)
    {
        return data[y][x];
    }
    // return coloumn
    std::vector<float>& operator()(int y)
    {
        return data[y];
    }

    void transpose()
    {
    }

private:
    std::vector<std::vector<float>> data;
};

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

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = i + 1; j < cols; j++) {
            std::swap(matrix[i][j], matrix[j][i]);
        }
    }
}

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

    for (int i = 1; i < M / 2; ++i)
        std::swap(vec[i], vec[new_indeces[i]]);

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

void FDHT2D(std::vector<std::vector<float>>* image_ptr)
{
    std::vector<std::vector<float>>& image = *image_ptr;
#ifdef PARALLEL
#pragma omp parallel for
#endif
    for (int i = 0; i < image.size(); ++i) {
        FDHT1D(&image[i]);
    }

    transpose(&image);

#ifdef PARALLEL
#pragma omp parallel for
#endif
    for (int i = 0; i < image.size(); ++i) {
        FDHT1D(&image[i]);
    }
}

/**
 * FHT3D(float ***CUBE, const size_t COLS) returns the multidimensional Hartley
 * transform of an 3-D array using a fast Hartley transform algorithm. The 3-D transform
 * is equivalent to computingthe 1-D transform along each dimension of CUBE.
 */
void DFHT3D(float*** cube, const int cols)
{
    if (cube == nullptr) {
        std::cout << "ERROR: FHT3D is not started. 3D array is NULL" << std::endl;
        return;
    }
    // Pre-work
    // FHT for 3rd axis
    const int log2 = (int)log2f(cols);

    // Indices for bit reversal operation
    std::vector<int> new_indeces(cols);
    bit_reverse(&new_indeces);

    // Main work
    // FHT by X
    for (int z = 0; z < cols; ++z) {
#ifdef PARALLEL
#pragma omp parallel for
#endif
        for (int y = 0; y < cols; ++y) {
            {
                // bitreverse swaping
                for (int x = 1; x < cols / 2; ++x)
                    std::swap(cube[z][y][x], cube[z][y][new_indeces[x]]);

                // butterfly
                for (int s = 1; s <= log2; ++s) {
                    int m = powf(2, s);
                    int m2 = m / 2;
                    int m4 = m / 4;

                    for (int r = 0; r <= cols - m; r = r + m) {
                        for (int j = 1; j < m4; ++j) {
                            int k = m2 - j;
                            float u = cube[z][y][r + m2 + j];
                            float v = cube[z][y][r + m2 + k];
                            float c = cosf((float)j * M_PI / (float)m2);
                            float s = sinf((float)j * M_PI / (float)m2);
                            cube[z][y][r + m2 + j] = u * c + v * s;
                            cube[z][y][r + m2 + k] = u * s - v * c;
                        }
                        for (int j = 0; j < m2; ++j) {
                            float u = cube[z][y][r + j];
                            float v = cube[z][y][r + j + m2];
                            cube[z][y][r + j] = u + v;
                            cube[z][y][r + j + m2] = u - v;
                        }
                    }
                }
            }
        }
    }

    // FHT by Y
    for (int z = 0; z < cols; ++z) {
#ifdef PARALLEL
#pragma omp parallel for
#endif
        for (int x = 0; x < cols; ++x) {
            {
                // bitreverse swaping
                for (int y = 1; y < cols / 2; ++y)
                    std::swap(cube[z][y][x], cube[z][new_indeces[y]][x]);

                // butterfly
                for (int s = 1; s <= log2; ++s) {
                    int m = powf(2, s);
                    int m2 = m / 2;
                    int m4 = m / 4;

                    for (int r = 0; r <= cols - m; r = r + m) {
                        for (int j = 1; j < m4; ++j) {
                            int k = m2 - j;
                            float u = cube[z][r + m2 + j][x];
                            float v = cube[z][r + m2 + k][x];
                            float c = cosf((float)j * M_PI / (float)m2);
                            float s = sinf((float)j * M_PI / (float)m2);
                            cube[z][r + m2 + j][x] = u * c + v * s;
                            cube[z][r + m2 + k][x] = u * s - v * c;
                        }
                        for (int j = 0; j < m2; ++j) {
                            float u = cube[z][r + j][x];
                            float v = cube[z][r + j + m2][x];
                            cube[z][r + j][x] = u + v;
                            cube[z][r + j + m2][x] = u - v;
                        }
                    }
                }
            }
        }
    }

    // FHT by Z
    for (int y = 0; y < cols; ++y) {
#ifdef PARALLEL
#pragma omp parallel for
#endif
        for (int x = 0; x < cols; ++x) {
            {
                // bitreverse swaping
                for (int z = 1; z < cols / 2; ++z)
                    std::swap(cube[z][y][x], cube[new_indeces[z]][y][x]);

                // butterfly
                for (int s = 1; s <= log2; ++s) {
                    int m = powf(2, s);
                    int m2 = m / 2;
                    int m4 = m / 4;

                    for (int r = 0; r <= cols - m; r = r + m) {
                        for (int j = 1; j < m4; ++j) {
                            int k = m2 - j;
                            float u = cube[r + m2 + j][y][x];
                            float v = cube[r + m2 + k][y][x];
                            float c = cosf((float)j * M_PI / (float)m2);
                            float s = sinf((float)j * M_PI / (float)m2);
                            cube[r + m2 + j][y][x] = u * c + v * s;
                            cube[r + m2 + k][y][x] = u * s - v * c;
                        }
                        for (int j = 0; j < m2; ++j) {
                            float u = cube[r + j][y][x];
                            float v = cube[r + j + m2][y][x];
                            cube[r + j][y][x] = u + v;
                            cube[r + j + m2][y][x] = u - v;
                        }
                    }
                }
            }
        }
    }
}
