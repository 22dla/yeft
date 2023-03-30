#include <rapidht.h>
#include <utilities.h>
#include <iostream>

int main() {
	// Define global 3D array sizes
	size_t rows = (size_t)pow(2, 5);

	auto a1 = make_data_1d<double>(rows);
	auto a2 = make_data_1d<double>(rows);
	//print_data_1d(a1);

	auto ptr = a1.data();

	double common_start, common_finish;
	common_start = clock() / static_cast<double>(CLOCKS_PER_SEC);

	RapiDHT::HartleyTransform ht(rows);
	//ht.mode = RapiDHT::GPU;
	
	ht.ForwardTransform(ptr);
	ht.InverseTransform(ptr);

	//print_data_1d(a1);

	common_finish = clock() / static_cast<double>(CLOCKS_PER_SEC);
	show_time(common_start, common_finish, "Common time");

	double sum = 0;
	for (int i = 0; i < rows; ++i) {
		sum += std::abs(a2[i] - ptr[i]);
		std::cout << i << " " << a2[i] << " " << ptr[i] << " " << std::abs(a2[i] - ptr[i]) << std::endl;
	}
	std::cout << "Error:\t" << sum << std::endl;

	return 0;
}