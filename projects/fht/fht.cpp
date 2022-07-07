#define HASH_SIZE 128
#define _USE_MATH_DEFINES
//#define MPIDataType MPI_UNSIGNED_CHAR
#define MPIDataType MPI_REAL
#define PARALLEL

#include <iostream>
#include <algorithm>
#include <mpi.h>
#include <bitset>
#include <math.h>
#include <vector>
#include <omp.h>
#include <assert.h>
#include "time.h"

//using DataType = unsigned __int8;
using DataType = float;
using namespace std;

void bitReverse(int *indices, const int length)
{
	const int log2n = (int)log2f(length);
	// array to store binary number
	bool *binaryNum = new bool[length];

	indices[0] = 0;
	for (int j = 1; j < length; ++j)
	{
		// counter for binary array
		int count = 0;
		int base = j;
		while (base > 0)
		{
			// storing remainder in binary array
			binaryNum[count] = base % 2;
			base /= 2;
			count++;
		}
		for (int i = count; i < log2n; ++i)
		{
			binaryNum[i] = 0;
		}

		int dec_value = 0;
		base = 1;
		for (int i = log2n - 1; i >= 0; i--)
		{
			if (binaryNum[i] == 1)
			{
				dec_value += base;
			}
			base = base * 2;
		}

		indices[j] = dec_value;
	}

	delete[] binaryNum;
}

void bitReverse(std::vector<size_t> &indices)
{
	const int log2n = (int)log2f(indices.size());
	// array to store binary number
	std::vector<bool>binaryNum(indices.size());

	indices[0] = 0;
	for (size_t j = 1; j < indices.size(); ++j)
	{
		// counter for binary array
		size_t count = 0;
		int base = (int)j;
		while (base > 0)
		{
			// storing remainder in binary array
			binaryNum[count] = (bool)base % 2;
			base /= 2;
			count++;
		}
		for (int i = count; i < log2n; ++i)
			binaryNum[i] = false;

		int dec_value = 0;
		base = 1;
		for (int i = log2n - 1; i >= 0; i--)
		{
			if (binaryNum[i])
			{
				dec_value += base;
			}
			base = base * 2;
		}

		indices[j] = dec_value;
	}
}

void fht(DataType *a, const int M)
{
	// FHT for 3rd axis
	const int log2 = (int)log2f(M);

	// Indices for bit reversal operation
	int *newIndeces = new int[M];

	bitReverse(newIndeces, M);

	for (int i = 1; i < M / 2; ++i)
	{
		std::swap(a[i], a[newIndeces[i]]);
	}

	for (int s = 1; s <= log2; ++s)
	{
		int m = powf(2, s);
		int m2 = m / 2;
		int m4 = m / 4;

		for (int r = 0; r <= M - m; r = r + m)
		{
			for (int j = 1; j < m4; ++j)
			{
				int k = m2 - j;
				float u = a[r + m2 + j];
				float v = a[r + m2 + k];
				float c = cosf((float)j * M_PI / (float)m2);
				float s = sinf((float)j * M_PI / (float)m2);
				a[r + m2 + j] = u * c + v * s;
				a[r + m2 + k] = u * s - v * c;
			}
			for (int j = 0; j < m2; ++j)
			{
				float u = a[r + j];
				float v = a[r + j + m2];
				a[r + j] = u + v;
				a[r + j + m2] = u - v;
			}
		}
	}

	delete[] newIndeces;
}

void fht(std::vector<DataType>& a)
{
	// FHT for 1rd axis
	size_t M = a.size();
	const int log2 = (int)log2f(M);
	const DataType m_pi = 3.14159265358979323846f;

	// Indices for bit reversal operation
	std::vector<size_t> newIndeces(M);
	bitReverse(newIndeces);

	for (int i = 1; i < M / 2; ++i)
		std::swap(a[i], a[newIndeces[i]]);

	for (int s = 1; s <= log2; ++s)
	{
		int m = powf(2, s);
		int m2 = m / 2;
		int m4 = m / 4;

		for (size_t r = 0; r <= M - m; r = r + m)
		{
			for (size_t j = 1; j < m4; ++j)
			{
				int k = m2 - j;
				DataType u = a[r + m2 + j];
				DataType v = a[r + m2 + k];
				DataType c = cosf((DataType)j * m_pi / (DataType)m2);
				DataType s = sinf((DataType)j * m_pi / (DataType)m2);
				a[r + m2 + j] = u * c + v * s;
				a[r + m2 + k] = u * s - v * c;
			}
			for (size_t j = 0; j < m2; ++j)
			{
				DataType u = a[r + j];
				DataType v = a[r + j + m2];
				a[r + j] = u + v;
				a[r + j + m2] = u - v;
			}
		}
	}

}

void initializeKernelHost(std::vector<DataType>& kernel, const int cols)
{
	const DataType m_pi = 3.14159265358979323846f;

	// Initialize matrices on the host
	for (size_t k = 0; k < cols; ++k) {
		for (size_t j = 0; j < cols; ++j){
			kernel[k*cols + j] = cosf(2 * m_pi*k*j / cols) + sinf(2 * m_pi*k*j / cols);
		}
	}
}

DataType* dht(DataType *a, const std::vector<DataType>& kernel, const int cols)
{
	DataType *result = new DataType[cols]();

	for (size_t i = 0; i < cols; i++)
		for (size_t j = 0; j < cols; j++)
			result[i] += (kernel[i*cols + j] * a[j]);

	return result;
}

/** 
* FHT3D(T ***CUBE, const size_t COLS) returns the multidimensional Hartley  
* transform of an 3-D array using a fast Hartley transform algorithm. The 3-D transform
* is equivalent to computingthe 1-D transform along each dimension of CUBE.
*/
template <typename T>
void FHT3D(T ***cube, const size_t cols)
{
	if (cube == nullptr)
	{
		std::cout << "ERROR: FHT3D is not started. 3D array is NULL" << std::endl;
		return;
	}
	// Pre-work
	// FHT for 3rd axis
	const int log2 = (int)log2f(cols);

	// Indices for bit reversal operation
	int *newIndeces = new int[cols];
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
				for (int s = 1; s <= log2; ++s)
				{
					int m = powf(2, s);
					int m2 = m / 2;
					int m4 = m / 4;

					for (int r = 0; r <= cols - m; r = r + m)
					{
						for (int j = 1; j < m4; ++j)
						{
							int k = m2 - j;
							float u = cube[z][y][r + m2 + j];
							float v = cube[z][y][r + m2 + k];
							float c = cosf((float)j * M_PI / (float)m2);
							float s = sinf((float)j * M_PI / (float)m2);
							cube[z][y][r + m2 + j] = u * c + v * s;
							cube[z][y][r + m2 + k] = u * s - v * c;
						}
						for (int j = 0; j < m2; ++j)
						{
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
				for (int s = 1; s <= log2; ++s)
				{
					int m = powf(2, s);
					int m2 = m / 2;
					int m4 = m / 4;

					for (int r = 0; r <= cols - m; r = r + m)
					{
						for (int j = 1; j < m4; ++j)
						{
							int k = m2 - j;
							float u = cube[z][r + m2 + j][x];
							float v = cube[z][r + m2 + k][x];
							float c = cosf((float)j * M_PI / (float)m2);
							float s = sinf((float)j * M_PI / (float)m2);
							cube[z][r + m2 + j][x] = u * c + v * s;
							cube[z][r + m2 + k][x] = u * s - v * c;
						}
						for (int j = 0; j < m2; ++j)
						{
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
				for (int s = 1; s <= log2; ++s)
				{
					int m = powf(2, s);
					int m2 = m / 2;
					int m4 = m / 4;

					for (int r = 0; r <= cols - m; r = r + m)
					{
						for (int j = 1; j < m4; ++j)
						{
							int k = m2 - j;
							float u = cube[r + m2 + j][y][x];
							float v = cube[r + m2 + k][y][x];
							float c = cosf((float)j * M_PI / (float)m2);
							float s = sinf((float)j * M_PI / (float)m2);
							cube[r + m2 + j][y][x] = u * c + v * s;
							cube[r + m2 + k][y][x] = u * s - v * c;
						}
						for (int j = 0; j < m2; ++j)
						{
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

/**
* init3Dcube(cube, cols) returns 
* initialized 3D cube with sizes COLS x COLS x COLS.
*/
template <typename T>
T *** init3Dcube(T ***arr, const size_t cols)
{
	// allocate memory
	arr = new T**[cols];
	for (size_t i = 0; i < cols; ++i)
	{
		arr[i] = new T*[cols];
		for (size_t j = 0; j < cols; ++j)
		{
			arr[i][j] = new T[cols];
		}
	}

	// filling arrays
	for (size_t j1 = 0; j1 < cols; ++j1)
	{
		//float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		T r = 1.0;
		for (size_t j2 = 0; j2 < cols; ++j2)
		{
			for (size_t j3 = 0; j3 < cols; ++j3)
			{
				arr[j1][j2][j3] = (cols + j1 + j2 + j3 + 1 + r) / cols;
				//cube[j1][j2][j3] = (j3 + cols*j2 + cols*cols*j1)/10.0;

				//std::cout << arr[j1][j2][j3] << "\t";
			}
			//std::cout << std::endl;
		}
		//std::cout << std::endl;
	}

	return arr;
}

/**
* clear3Dcube(cube, cols) clears
* 3D cube with sizes COLS x COLS x COLS.
*/
template <typename T>
void clear3Dcube(T ***arr, const size_t cols)
{
	if (arr == nullptr) {
		std::cout << "Warning: Clearing is not started. 3D array is NULL" << std::endl;
		return;
	}

	for (int i = 0; i < cols; ++i) {
		for (int j = 0; j < cols; ++j)
			delete[] arr[i][j];
		delete[] arr[i];
	}
	delete[] arr;
}

int main()
{
	// Initialize the MPI environment
	MPI_Init(NULL, NULL);

	// Define global 3D array sizes
	const size_t cols = pow(2, 13);

	// input data
	//DataType *a = new DataType[cols];
	std::vector<DataType> a1(cols);
	std::vector<std::vector<DataType>> a2(cols);
	for (size_t j3 = 0; j3 < cols; ++j3)
	{
		a2[j3].resize(cols);
		a1[j3] = (cols + j3 + 2) / cols;
		for (size_t j4 = 0; j4 < cols; ++j4)
			a2[j3][j4] = (cols + j3 + j4 + 2) / cols;
	}


	DataType*** cube;
	//cube = init3Dcube(cube, cols);

	clock_t commonStart, commonStop;
	commonStart = clock();
	auto start1 = MPI_Wtime();

	//FHT3D(cube, cols);
	for (int i0 = 0; i0 < 2; ++i0)
#ifdef PARALLEL
#pragma omp parallel for
#endif	
		for (int i = 0; i < cols; ++i)
			fht(a2[i]);

	auto finish1 = MPI_Wtime();
	commonStop = clock();

	printf("FHT Time:\t%3.5f sec \n", finish1 - start1);
	double time_taken = double(commonStop - commonStart) / double(CLOCKS_PER_SEC);
	printf("Common time:\t%3.5f sec \n", time_taken);
	
	// Finalize the MPI environment.
	MPI_Finalize();

	// Deleting memories
	//clear3Dcube(cube, cols);
	return 0;
}
