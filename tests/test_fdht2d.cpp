#include <iostream>
#include <rapidht.h>
#include <utilities.h>

int main(int argc, char** argv) {
	// Define global 3D array sizes
	int rows = (int)pow(2, 3);
	int cols = rows;
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

	auto a3 = make_data<double>({ rows, cols });

	double common_start, common_finish;
	common_start = clock() / static_cast<double>(CLOCKS_PER_SEC);

	auto ptr = a3.data();

	print_data_2d(ptr, rows, cols);

	RapiDHT::HartleyTransform ht(rows, cols, 0, mode);
	ht.ForwardTransform(ptr);

	print_data_2d(ptr, rows, cols);

	common_finish = clock() / static_cast<double>(CLOCKS_PER_SEC);
	show_time(common_start, common_finish, "Common time");
	return 0;
}