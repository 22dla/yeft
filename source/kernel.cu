#include "kernel.h"
#include "device_launch_parameters.h"
#include "dev_array.h"

__global__ void matrixMultiplicationKernel(const double* A, const double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Переменная для хранения значения элемента результирующей матрицы
    double value = 0;

    // Проверяем, что индекс находится в пределах матрицы
    if (row < N && col < N) {
        // Выполняем суммирование произведений соответствующих элементов строк и столбцов
        for (int k = 0; k < N; ++k) {
            value += A[row * N + k] * B[k * N + col];
        }
        // Записываем результат в результирующую матрицу
        C[row * N + col] = value;
    }
}

__global__ void matrixVectorMultKernel(const double* A, const double* x, double* y, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		double sum = 0.0f;
		for (int j = 0; j < N; j++) {
			sum += A[i * N + j] * x[j];
		}
		y[i] = sum;
	}
}

__global__ void matrixTransposeKernel(double* A, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < N && j < N && i < j) {
		double tmp = A[i * N + j];
		A[i * N + j] = A[j * N + i];
		A[j * N + i] = tmp;
	}
}

__global__ void BracewellTransform2DKernel(double* d_image, size_t cols, size_t rows) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rows && j < cols) {
        double A = d_image[i * cols + j];
        double B = (i > 0 && j > 0) ? d_image[i * cols + (cols - j)] : A;
        double C = (i > 0 && j > 0) ? d_image[(rows - i) * cols + j] : A;
        double D = (i > 0 && j > 0) ? d_image[(rows - i) * cols + (cols - j)] : A;
        d_image[i * cols + j] = (A + B + C - D) / 2.0;
    }
}

void matrixMultiplication(const double *A, const double *B, double *C, int N) {
	dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Запуск ядра на GPU
    matrixMultiplicationKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
}

void vectorMatrixMultiplication(const double* A, const double* x, double* y, int N) {

	int threadsPerBlock, blocksPerGrid;
	threadsPerBlock = (N > 512) ? 512 : N;
	blocksPerGrid = (N > 512) ? (N + threadsPerBlock - 1) / threadsPerBlock : 1;
	matrixVectorMultKernel <<< blocksPerGrid, threadsPerBlock >>> (A, x, y, N);
}

void matrixTranspose(double* A, int N) {
	// declare the number of blocks per grid and the number of threads per block
	// use 1 to 512 threads per block
	dim3 threadsPerBlock(N, N);
	dim3 blocksPerGrid(1, 1);
	if (N * N > 512) {
		threadsPerBlock.x = 512;
		threadsPerBlock.y = 512;
		//blocksPerGrid.x = ceil(double(N) / double(threadsPerBlock.x));
		//blocksPerGrid.y = ceil(double(N) / double(threadsPerBlock.y));

		blocksPerGrid.x = (N + threadsPerBlock.x - 1) / threadsPerBlock.x;
		blocksPerGrid.y = (N + threadsPerBlock.y - 1) / threadsPerBlock.y;
	}

	matrixTransposeKernel <<< blocksPerGrid, threadsPerBlock >> > (A, N);
}

void BracewellTransform2D(double* d_image, int cols, int rows) {
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((cols + 31) / 32, (rows + 31) / 32);

    BracewellTransform2DKernel<<<blocksPerGrid, threadsPerBlock>>>(d_image, cols, rows);
}
