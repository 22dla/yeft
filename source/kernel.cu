#include "kernel.h"
#include "device_launch_parameters.h"
#include "dev_array.h"

#define TILE_SIZE 16  // Размер блока (мозаики)

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
    __shared__ double tile[TILE_SIZE][TILE_SIZE];

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    int idx = y * N + x;

    if (x < N && y < N) {
        tile[threadIdx.y][threadIdx.x] = A[idx];
    }
    __syncthreads();

    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    idx = y * N + x;

    if (x < N && y < N) {
        A[idx] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void BracewellTransform2DKernel(double* d_image, size_t cols, size_t rows) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rows && j < cols) {
        int top_left = i * cols + j;
        int top_right = i * cols + (cols - j - 1);
        int bottom_left = (rows - i - 1) * cols + j;
        int bottom_right = (rows - i - 1) * cols + (cols - j - 1);

        double A = d_image[top_left];
        double B = (i < rows && j < cols) ? d_image[top_right] : A;
        double C = (i < rows && j < cols) ? d_image[bottom_left] : A;
        double D = (i < rows && j < cols) ? d_image[bottom_right] : A;

        d_image[top_left] = (A + B + C - D) / 2.0;
    }
}

void matrixMultiplication(const double *A, const double *B, double *C, int N) {
	dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
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

void BracewellTransform2D(double* d_image, int N) {
	int threadsPerBlock, blocksPerGrid;
	threadsPerBlock = (N > 512) ? 512 : N;
	blocksPerGrid = (N > 512) ? (N + threadsPerBlock - 1) / threadsPerBlock : 1;

    BracewellTransform2DKernel<<<blocksPerGrid, threadsPerBlock>>>(d_image, N, N);
}
