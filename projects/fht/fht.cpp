#define HASH_SIZE 128
#define _USE_MATH_DEFINES
//#define MPIDataType MPI_UNSIGNED_CHAR
#define MPIDataType MPI_REAL

#include <iostream>
#include <algorithm>
#include <mpi.h>
#include <bitset>
#include <math.h>
#include <vector>
#include <omp.h>

//using DataType = unsigned __int8;
using DataType = float;

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
	// FHT for 3rd axis
	size_t M = a.size();
	const int log2 = (int)log2f(M);
	const DataType m_pi = 3.14159265358979323846f;

	// Indices for bit reversal operation
	std::vector<size_t> newIndeces(M);
	bitReverse(newIndeces);

	for (int i = 1; i < M / 2; ++i)
	{
		std::swap(a[i], a[newIndeces[i]]);
	}

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

using namespace std;

int main()
{
	// Initialize the MPI environment
	MPI_Init(NULL, NULL);

	// Define global 3D array sizes
	const int cols = pow(2, 9);

	DataType *a = new DataType[cols];

	// input data
	for (int j = 0; j < cols; ++j)
	{
		//float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		float r = 1.0f;

		a[j] = (cols + j + 1 + r) / cols;
	}

	// FHT
	auto start1 = MPI_Wtime();
	for (int direction = 0; direction < 3; ++direction) {
		for (int i = 0; i < cols; ++i) {
#pragma omp parallel for
			for (int j = 0; j < cols; ++j) {
				fht(a, cols);
			}
		}
	}

	auto finish1 = MPI_Wtime();
	std::cout << std::endl << "FHT Time = " << finish1 - start1 << " seconds." << std::endl;

	// Deleting memories
	delete[] a;

	// Finalize the MPI environment.
	MPI_Finalize();
	return 0;
}
