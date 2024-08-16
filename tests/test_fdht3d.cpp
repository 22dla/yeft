#include <iostream>
#include <rapidht.h>
#include <utilities.h>
#include <cmath>
#include <cstring>

int main(int argc, char** argv) {
	// Define global 3D array sizes
	int rows = static_cast<int>(pow(2, 9));
	int cols = rows;
	int depth = cols;
	RapiDHT::Modes mode = RapiDHT::GPU;

	// If arguments are parced then exactly two arguments are required
	if (argc >= 2) {
		rows = std::atoi(argv[1]);
		cols = rows;
		depth = rows;
		if (argc >= 3) {
			auto device = argv[2];
			if (!strcmp(device, "CPU")) {
				mode = RapiDHT::CPU;
			}
			else if (!strcmp(device, "GPU")) {
				mode = RapiDHT::GPU;
			}
			else if (!strcmp(device, "RFFT")) {
				mode = RapiDHT::RFFT;
			}
			else {
				std::cerr << "Error: device must be either CPU, GPU or RFFT" << std::endl;
				return 1;
			}
		}
	}

	std::cout << "Cols: " << cols << std::endl;
	std::cout << "Rows: " << rows << std::endl;
	std::cout << "Depth: " << depth << std::endl;
	std::cout << "Mode: " << (mode == RapiDHT::CPU ? "CPU" :
		mode == RapiDHT::GPU ? "GPU" : "RFFT") << std::endl << std::endl;

	auto common_start = std::chrono::high_resolution_clock::now();

	std::cout << "making data...";
	auto making_start = std::chrono::high_resolution_clock::now();
	auto a3 = make_data<double>({ rows, cols, depth });
	auto making_finish = std::chrono::high_resolution_clock::now();
	auto making_time = std::chrono::duration_cast<std::chrono::milliseconds>(making_finish - making_start);
	std::cout << "time:\t" << making_time.count() / 1000.0 << std::endl;

	std::cout << "HT calculation...";
	auto calculation_start = std::chrono::high_resolution_clock::now();
	auto ptr = a3.data();
	RapiDHT::HartleyTransform ht(rows, cols, depth, mode);
	
	ht.ForwardTransform(ptr);

	auto calculation_finish = std::chrono::high_resolution_clock::now();
	auto calculation_time = std::chrono::duration_cast<std::chrono::milliseconds>(calculation_finish - calculation_start);
	std::cout << "time:\t" << calculation_time.count() / 1000.0 << std::endl;

	auto common_finish = std::chrono::high_resolution_clock::now();
	auto common_time = std::chrono::duration_cast<std::chrono::milliseconds>(common_finish - common_start);
	std::cout << "common time:\t" << common_time.count() / 1000.0 << std::endl;
	return 0;
}
