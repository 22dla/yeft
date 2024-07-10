#include <iostream>
#include <rapidht.h>
#include <utilities.h>
#include <cmath>
#include <cstring>

int main(int argc, char** argv) {
	// Define global 3D array sizes
	int rows = static_cast<int>(pow(2, 10));
	int cols = rows;
	int images_num = 10;
	RapiDHT::Modes mode = RapiDHT::CPU;

	// If arguments are parced then exactly two arguments are required
	if (argc >= 2) {
		if (argc >= 3) {
			rows = std::atoi(argv[1]);
			cols = rows;
			images_num = std::atoi(argv[2]);
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
			std::cerr << "Usage: " << argv[0] << " rows images_num" << std::endl;
			return 1;
		}
	}

	auto common_start = std::chrono::high_resolution_clock::now();

	std::cout << "making data...";
	auto making_start = std::chrono::high_resolution_clock::now();
	auto a3 = makeData<double>({ rows, cols, images_num });
	auto making_finish = std::chrono::high_resolution_clock::now();
	auto making_time = std::chrono::duration_cast<std::chrono::milliseconds>(making_finish - making_start);
	std::cout << "time:\t" << making_time.count() / 1000.0 << std::endl;

	std::cout << "HT calculation...";
	auto calculation_start = std::chrono::high_resolution_clock::now();
	auto ptr = a3.data();
	RapiDHT::HartleyTransform ht(rows, cols, 0, mode);
	for (int i = 0; i < images_num; ++i) {
		//printData2D(ptr + i * cols * rows, rows, cols);
		ht.ForwardTransform(ptr + i * cols * rows);
		//printData2D(ptr + i * cols * rows, rows, cols);
	}
	auto calculation_finish = std::chrono::high_resolution_clock::now();
	auto calculation_time = std::chrono::duration_cast<std::chrono::milliseconds>(calculation_finish - calculation_start);
	std::cout << "time:\t" << calculation_time.count() / 1000.0 << std::endl;

	auto common_finish = std::chrono::high_resolution_clock::now();
	auto common_time = std::chrono::duration_cast<std::chrono::milliseconds>(common_finish - common_start);
	std::cout << "common time:\t" << common_time.count() / 1000.0 << std::endl;
	return 0;
}
