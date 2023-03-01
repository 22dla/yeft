#include <fht/fht.h>

int main()
{
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Define global 3D array sizes
    const size_t cols = pow(2, 4);

    // input data
    std::vector<DataType> a1(cols);
    std::vector<std::vector<DataType>> a2(cols);
    for (size_t j3 = 0; j3 < cols; ++j3) {
        a2[j3].resize(cols);
        a1[j3] = static_cast<DataType>(20 * cols + j3 + 2) / cols;
        for (size_t j4 = 0; j4 < cols; ++j4)
            a2[j3][j4] = static_cast<DataType>(cols + j3 + j4 + 2) / cols;
    }

    double kernel_start, kernel_finish, common_start, common_finish;
    common_start = clock() / static_cast<double>(CLOCKS_PER_SEC);
    kernel_start = MPI_Wtime();

    std::vector<DataType> kernel;
    initializeKernelHost(&kernel, a2[0].size());

    std::vector<DataType> test;
    // DataType* test;
    for (int i0 = 0; i0 < 50; ++i0) {
        for (int direction = 0; direction < 2; ++direction) {
#ifdef PARALLEL
#pragma omp parallel for
#endif
            for (int i = 0; i < cols; ++i) {
                // fht1d(&a2[i]);
                test = dht1d(a2[i], kernel);
            }
        }
    }

    kernel_finish = MPI_Wtime();
    common_finish = clock() / static_cast<double>(CLOCKS_PER_SEC);

    showTime(kernel_start, kernel_finish, "FHT time");
    showTime(common_start, common_finish, "Common time");
    writeData(a1, std::ios_base::out, "input");
    writeData(test, std::ios_base::app, "output");

    // Finalize the MPI environment.
    MPI_Finalize();
    return 0;
}
