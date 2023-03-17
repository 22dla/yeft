#include <cufht.h>

int main(int argc, char* argv[])
{
    // Define global 3D array sizes
    size_t cols = pow(2, 9);
    size_t image_num = 100;

    if (argc > 3) {
        printf("Only R (resolution) and N (number of images)\n");
        return 1;
    } else if (argc == 3) {
        sscanf(argv[1], "%d", &cols);
        sscanf(argv[2], "%d", &image_num);
    } else if (argc == 2) {
        printf("Standart image num = 100\n");
        return 1;
    }
    const size_t rows = cols;

    std::vector<DataType> h_B(cols * cols * image_num);
    std::vector<DataType> h_C(cols * cols);

    // input data
    for (int j1 = 0; j1 < cols * cols * image_num; ++j1) {
        // float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        float r = 1.0f;
        h_B[j1] = (cols + j1 + 1 + r) / (cols * cols * image_num);
    }

    float time1;
    clock_t commonStart, commonStop;
    cudaEvent_t start, stop;
    commonStart = clock();

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // DHT
    HT2DCuda(h_B, h_C, cols, image_num);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time1, start, stop);
    commonStop = clock();

    printf("GPU Time:  \t%3.3f sec \n", time1 / 1000.0);
    double time_taken = double(commonStop - commonStart) / double(CLOCKS_PER_SEC);
    printf("Common time:  \t%3.3f sec \n", time_taken);
    return 0;
}