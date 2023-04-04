#include <rapidht.h>
#include <utilities.h>

int main() {
	// Define global 3D array sizes
	int cols = (int)pow(2, 2);
	int rows = cols;

	auto a3 = make_data<double>({ cols, rows });

	double common_start, common_finish;
	common_start = clock() / static_cast<double>(CLOCKS_PER_SEC);

	auto ptr = a3.data();

	print_data_2d(ptr, rows, cols);

	RapiDHT::HartleyTransform ht(cols, rows);
	ht.mode = RapiDHT::Modes::GPU;
	ht.ForwardTransform(ptr);

	print_data_2d(ptr, rows, cols);

	common_finish = clock() / static_cast<double>(CLOCKS_PER_SEC);
	show_time(common_start, common_finish, "Common time");
	return 0;
}