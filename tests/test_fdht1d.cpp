#include <rapidht.h>
#include <utilities.h>
#include <iostream>
#include <cmath>
#include <numeric>
#include <cstring>

int main(int argc, char** argv) {
	int rows = static_cast<int>(pow(2, 14));
	RapiDHT::Modes mode = RapiDHT::GPU;

	// If arguments is parced then exactly one argument is required
	if (argc >= 2) {
		rows = std::atoi(argv[1]);

		if (argc >= 3) {
			auto device = argv[2];

			if (!strcmp(device, "CPU")) {
				mode = RapiDHT::CPU;
			} else if (!strcmp(device, "GPU")) {
				mode = RapiDHT::GPU;
			} else if (!strcmp(device, "RFFT")) {
				mode = RapiDHT::RFFT;
			} else {
				std::cerr << "Error: device must be either CPU, GPU or RFFT" << std::endl;
				return 1;
			}
		}
		if (argc >= 4) {
			std::cerr << "Usage: " << argv[0] << " rows" << std::endl;
			return 1;
		}
	}

	auto a1_1 = make_data<double>({ rows });
	auto a1_2 = a1_1;
	//print_data_1d(a1_1.data(), rows);

	auto ptr = a1_1.data();

	double common_start, common_finish;
	common_start = clock() / static_cast<double>(CLOCKS_PER_SEC);

	RapiDHT::HartleyTransform ht(rows, 0, 0, mode);
	
	ht.ForwardTransform(ptr);
	ht.InverseTransform(ptr);

	//print_data_1d(a1_1.data(), rows);

	common_finish = clock() / static_cast<double>(CLOCKS_PER_SEC);
	show_time(common_start, common_finish, "Common time");

	double sum_sqr = std::transform_reduce(
		a1_1.begin(), a1_1.end(), a1_2.begin(), 0.0, std::plus<>(),
		[](double x, double y) { return (x - y)*(x - y); }
	);
	std::cout << "Error:\t" << std::sqrt(sum_sqr) << std::endl;

	return 0;
}
