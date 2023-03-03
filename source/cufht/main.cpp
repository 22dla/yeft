#include <cufht.h>

int main()
{
    // Define global ND array sizes
    const int cols = pow(2, 11);

    std::vector<DataType> h_B(cols * cols);
    std::vector<DataType> h_C(cols * cols);

    // input data
    for (int j1 = 0; j1 < cols * cols; ++j1) {
        // float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        float r = 1.0f;
        h_B[j1] = (cols + j1 + 1 + r) / cols;
    }

    float time1;
    clock_t commonStart, commonStop;
    cudaEvent_t start, stop;
    commonStart = clock();

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // DHT
    HT2DCuda(h_B, h_C, cols);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time1, start, stop);
    commonStop = clock();

    printf("GPU Time:  \t%3.3f sec \n", time1 / 1000.0);
    double time_taken = double(commonStop - commonStart) / double(CLOCKS_PER_SEC);
    printf("Common time:  \t%3.3f sec \n", time_taken);
    return 0;
    return 0;
}