#include <rapidht.h>
#include <utilities.h>
#include <iostream>
#include <cmath>
#include <numeric>
#include <cstring>

int main(int argc, char** argv) {
	size_t rows = static_cast<size_t>(pow(2, 19));
	RapiDHT::Modes mode = RapiDHT::CPU;

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

	auto a1_1 = makeData<double>({ rows });
	auto a1_2 = makeData<double>({ rows });
	//printData1D(a1);

	double common_start, common_finish;
	common_start = clock() / static_cast<double>(CLOCKS_PER_SEC);

	RapiDHT::HartleyTransform ht(rows, 0, 0, mode);
	
	ht.ForwardTransform(a1_1);
	ht.InverseTransform(a1_1);

	//printData1D(a1);

	common_finish = clock() / static_cast<double>(CLOCKS_PER_SEC);
	showTime(common_start, common_finish, "Common time");

	double sum = std::transform_reduce(
		a1_1.begin(), a1_1.end(), a1_2.begin(), 0.0,
		std::plus<>(),
		[](double x, double y) { return std::abs(x - y); }
	);
	std::cout << "Error:\t" << sum << std::endl;

	return 0;
}
