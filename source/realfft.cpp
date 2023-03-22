#include <iostream>

#include <realfft.h>

void showTime(double startTime, double finishTime, std::string message)
{
    std::cout << message + ":\t" << finishTime - startTime << " sec" << std::endl;
}

void fft1dv1(std::vector<Complex>* x)
{
    // DFT
    unsigned int N = (*x).size(), k = N, n;
    double thetaT = 3.14159265358979323846264338328L / N;
    Complex phiT = Complex(cos(thetaT), -sin(thetaT)), T;
    while (k > 1) {
        n = k;
        k >>= 1;
        phiT = phiT * phiT;
        T = 1.0L;
        for (unsigned int l = 0; l < k; l++) {
            for (unsigned int a = l; a < N; a += n) {
                unsigned int b = a + k;
                Complex t = (*x)[a] - (*x)[b];
                (*x)[a] += (*x)[b];
                (*x)[b] = t * T;
            }
            T *= phiT;
        }
    }
    // Decimate
    unsigned int m = (unsigned int)log2(N);
    for (unsigned int a = 0; a < N; a++) {
        unsigned int b = a;
        // Reverse bits
        b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
        b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
        b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
        b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
        b = ((b >> 16) | (b << 16)) >> (32 - m);
        if (b > a) {
            Complex t = (*x)[a];
            (*x)[a] = (*x)[b];
            (*x)[b] = t;
        }
    }
}

void fft1dv2(std::vector<Complex>* x)
{
    int n = (*x).size();

    // Bit-reverse permutation
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        while (j >= bit) {
            j -= bit;
            bit >>= 1;
        }
        j += bit;
        if (i < j) {
            std::swap((*x)[i], (*x)[j]);
        }
    }

    // FFT
    for (int k = 2; k <= n; k <<= 1) {
        double angle = 2 * M_PI / k;
        Complex wk(cos(angle), sin(angle));
        for (int i = 0; i < n; i += k) {
            Complex w(1);
            for (int j = 0; j < k / 2; ++j) {
                Complex t = w * (*x)[i + j + k / 2];
                Complex u = (*x)[i + j];
                (*x)[i + j] = u + t;
                (*x)[i + j + k / 2] = u - t;
                w *= wk;
            }
        }
    }
}
