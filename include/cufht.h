/**
* DHT1DCuda(double* vector, const int length) returns the Hartley
* transform of an 1D array using a matrix x vector multiplication.
*/
void DHT1DCuda(double* vector, double* vandermone_matrix, const int length);

void DHT2DCuda(double* matrix, double* vandermone_matrix, const int rows, const int cols);
