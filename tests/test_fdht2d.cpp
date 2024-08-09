#include <iostream>
#include <rapidht.h>
#include <utilities.h>
#include <cmath>
#include <cstring>

int main(int argc, char** argv) {
	// Define global 3D array sizes
	size_t rows = static_cast<int>(pow(2, 3));
	size_t cols = rows;
	RapiDHT::Modes mode = RapiDHT::CPU;

	// If arguments are parced then exactly two arguments are required
	if (argc >= 2) {
		if (argc >= 3) {
			rows = std::atoi(argv[1]);
			cols = std::atoi(argv[2]);
			if (argc >= 4) {
				auto device = argv[3];
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
		} else {
			std::cerr << "Usage: " << argv[0] << " rows cols" << std::endl;
			return 1;
		}
	}

	auto a2 = makeData<double>({ rows, cols });

	double common_start, common_finish;
	common_start = clock() / static_cast<double>(CLOCKS_PER_SEC);

	printData2D(a2.data(), rows, cols);

	RapiDHT::HartleyTransform ht(rows, cols, 0, mode);
	ht.ForwardTransform(a2);

	printData2D(a2.data(), rows, cols);

	common_finish = clock() / static_cast<double>(CLOCKS_PER_SEC);
	showTime(common_start, common_finish, "Common time");
	return 0;
}
