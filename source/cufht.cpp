#include <cufht.h>

void initializeKernelHost(std::vector<DataType>& kernel, const int cols)
{
    const DataType m_pi = 3.14159265358979323846f;

    // Initialize matrices on the host
    for (size_t k = 0; k < cols; ++k) {
        for (size_t j = 0; j < cols; ++j) {
            kernel[k * cols + j] = cosf(2 * m_pi * k * j / cols) + sinf(2 * m_pi * k * j / cols);
        }
    }
}

// template <typename T>
void HT2DCuda(const std::vector<DataType>& X, std::vector<DataType>& Y, const int cols, const int image_num)
{
    // Allocate memory on the host
    std::vector<DataType> h_A(cols * cols);

    // Allocate memory on the device
    dev_array<DataType> d_A(cols * cols); // matrix for one line
    dev_array<DataType> d_X(cols * cols); // one slice
    dev_array<DataType> d_Y(cols * cols); // one slice

    // Initialize matrices on the host
    initializeKernelHost(h_A, cols);
    // transfer CPU -> GPU
    d_A.set(&h_A[0], cols * cols);

    for (int i0 = 0; i0 < image_num; ++i0) {
        for (int direction = 0; direction < 2; ++direction) {
            // transfer CPU -> GPU
            d_X.set(&X[i0 * cols * cols], cols * cols);
            matrixMultiplication(d_A.getData(), d_X.getData(), d_Y.getData(), cols);
            // transfer GPU -> CPU
            d_Y.get(&Y[0], cols * cols);
            cudaDeviceSynchronize();
        }
    }

    cudaDeviceSynchronize();
}
