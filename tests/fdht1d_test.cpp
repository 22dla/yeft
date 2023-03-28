#include <rapidht.h>
#include <utilities.h>
#include <iostream>

int main() {
	// Define global 3D array sizes
	size_t rows = (size_t)pow(2, 9);

	auto a1 = make_data_1d<float>(rows);
	auto a2 = make_data_1d<float>(rows);
	//print_data_1d(a1);

	auto ptr = a1.data();

	double common_start, common_finish;
	common_start = clock() / static_cast<double>(CLOCKS_PER_SEC);

	RapiDHT::HartleyTransform ht(rows);
	RapiDHT::HartleyTransform htCuda(rows, 0, 0, RapiDHT::GPU);
	//ht.ForwardTransform(ptr);
	htCuda.ForwardTransform(ptr);
	htCuda.InverseTransform(ptr);

	//print_data_1d(a1);

	common_finish = clock() / static_cast<double>(CLOCKS_PER_SEC);
	show_time(common_start, common_finish, "Common time");

	float sum = 0;
	for (int i = 0; i < rows; ++i) {
		sum += std::abs(a2[i] - ptr[i]);
	}
	std::cout << "Error:\t" << sum << std::endl;

	return 0;
}