#include <fftw3.h>
#include <iostream>

void PrintArray(const fftw_complex* data, const int N)
{
    std::cout << std::endl
              << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << "(" << data[i][0] << ", " << data[i][1] << ")"
                  << "\t";
    }
}

template<typename T>
void ShowTime(T startTime, T finishTime, std::string message)
{
    std::cout << message + ":\t" << finishTime - startTime << " sec" << std::endl;
}

int main()
{
    fftw_complex *in, *out;
    fftw_plan p;

    int N = 32;

    in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);

    for (int i = 0; i < N; ++i) {
        *in[i] = static_cast<double>(N - i + 1) * sinf(i + 1) / N;
    }
    // PrintArray(in, N);

    p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    double kernel_start, kernel_finish, common_start, common_finish;
    common_start = clock() / static_cast<double>(CLOCKS_PER_SEC);
    int count = 0;
    for (int i = 0; i < 200; ++i) {
        for (int k = 0; k < 2*N; ++k) {
            fftw_execute(p); /* repeat as needed */
            count++;
        }
    }

    common_finish = clock() / static_cast<double>(CLOCKS_PER_SEC);
    ShowTime(common_start, common_finish, "Common time");

    PrintArray(out, N);

    std::cout << count << std::endl;
    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);

    return 0;
}