#include <fht.h>

#include <fstream>
#include <iostream>

void bitReverse(std::vector<size_t>* indices)
{
    const int kLog2n = static_cast<int>(log2f((*indices).size()));

    // array to store binary number
    std::vector<bool> binary_num((*indices).size());

    (*indices)[0] = 0;
    for (int j = 1; j < (*indices).size(); ++j) {
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
        (*indices)[j] = dec_value;
    }
}

void fht1d(std::vector<DataType>* a)
{
    // FHT for 1rd axis
    size_t M = (*a).size();
    const int kLog2n = (int)log2f(M);
    const DataType kPi = 3.14159265358979323846f;

    // Indices for bit reversal operation
    std::vector<size_t> new_indeces(M);
    bitReverse(&new_indeces);

    for (int i = 1; i < M / 2; ++i)
        std::swap((*a)[i], (*a)[new_indeces[i]]);

    for (int s = 1; s <= kLog2n; ++s) {
        int m = powf(2, s);
        int m2 = m / 2;
        int m4 = m / 4;

        for (size_t r = 0; r <= M - m; r = r + m) {
            for (size_t j = 1; j < m4; ++j) {
                int k = m2 - j;
                DataType u = (*a)[r + m2 + j];
                DataType v = (*a)[r + m2 + k];
                DataType c = cosf(static_cast<DataType>(j) * kPi / m2);
                DataType s = sinf(static_cast<DataType>(j) * kPi / m2);
                (*a)[r + m2 + j] = u * c + v * s;
                (*a)[r + m2 + k] = u * s - v * c;
            }
            for (size_t j = 0; j < m2; ++j) {
                DataType u = (*a)[r + j];
                DataType v = (*a)[r + j + m2];
                (*a)[r + j] = u + v;
                (*a)[r + j + m2] = u - v;
            }
        }
    }
}

void initializeKernelHost(std::vector<DataType>* kernel, const int cols)
{
    const DataType kPi = 3.14159265358979323846f;
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

std::vector<DataType> dht1d(const std::vector<DataType>& a, const std::vector<DataType>& kernel)
{
    std::vector<DataType> result(a.size());

    for (size_t i = 0; i < a.size(); i++)
        for (size_t j = 0; j < a.size(); j++)
            result[i] += (kernel[i * a.size() + j] * a[j]);

    // RVO works
    return result;
}

void showTime(double startTime, double finishTime, std::string message)
{
    std::cout << message + ":\t" << finishTime - startTime << " sec" << std::endl;
}

template <typename T>
void writeData(const std::vector<T>& vec, int mode,
    const std::string& name, const std::string& path)
{
    std::fstream file;
    file.open(path, mode);

    switch (mode) {
    case std::ios_base::out: {

        file << ";";
        for (int i = 0; i < vec.size(); ++i) {
            file << i << ";";
        }

        file << std::endl
             << name << ";";
        for (int i = 0; i < vec.size(); ++i) {
            file << vec[i] << ";";
        }
        break;
    }
    case std::ios_base::app: {
        file << std::endl
             << name << ";";
        for (int i = 0; i < vec.size(); ++i) {
            file << vec[i] << ";";
        }
        break;
    }
    default:
        break;
    }

    file.close();
}

/**
 * FHT3D(T ***CUBE, const size_t COLS) returns the multidimensional Hartley
 * transform of an 3-D array using a fast Hartley transform algorithm. The 3-D transform
 * is equivalent to computingthe 1-D transform along each dimension of CUBE.
 */
template <typename T>
void FHT3D(T*** cube, const size_t cols)
{
    if (cube == nullptr) {
        std::cout << "ERROR: FHT3D is not started. 3D array is NULL" << std::endl;
        return;
    }
    // Pre-work
    // FHT for 3rd axis
    const int log2 = (int)log2f(cols);

    // Indices for bit reversal operation
    int* newIndeces = new int[cols];
    bitReverse(newIndeces, cols);

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
                    std::swap(cube[z][y][x], cube[z][y][newIndeces[x]]);

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
                    std::swap(cube[z][y][x], cube[z][newIndeces[y]][x]);

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
                    std::swap(cube[z][y][x], cube[newIndeces[z]][y][x]);

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

    // Deleting memories
    delete[] newIndeces;
}
