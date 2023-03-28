#include <rapidht.h>
#include <utilities.h>

int main() {
	// Define global 3D array sizes
	size_t rows = (size_t)pow(2, 4);

	auto a1 = make_data_1d<float>(rows);
	print_data_1d(a1);

	auto ptr = a1.data();

	RapiDHT::HartleyTransform ht(rows);
	ht.ForwardTransform(ptr);
	ht.InverseTransform(ptr);

	print_data_1d(a1);

	return 0;
}