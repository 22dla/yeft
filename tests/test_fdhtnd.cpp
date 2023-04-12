#include <iostream>
#include <rapidht.h>
#include <utilities.h>
#include <cmath>
#include <cstring>

int main(int argc, char** argv) {
	// Define global 3D array sizes
	int rows = static_cast<int>(pow(2, 8));
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

	double making_start, making_finish, 
		calculation_start, calculation_finish, 
		common_start, common_finish = 0;
	common_start = clock() / static_cast<double>(CLOCKS_PER_SEC);


	std::cout << "making data...";
	making_start = clock() / static_cast<double>(CLOCKS_PER_SEC);
	auto a3 = make_data<double>({ rows, cols, images_num });
	making_finish = clock() / static_cast<double>(CLOCKS_PER_SEC);
	show_time(making_start, making_finish, "time");

	std::cout << "HT calculation...";
	calculation_start = clock() / static_cast<double>(CLOCKS_PER_SEC);
	auto ptr = a3.data();
	RapiDHT::HartleyTransform ht(rows, cols, 0, mode);
	for (int i = 0; i < images_num; ++i) {
		//print_data_2d(ptr + i * cols * rows, rows, cols);
		ht.ForwardTransform(ptr + i * cols * rows);
		//print_data_2d(ptr + i * cols * rows, rows, cols);
	}
	calculation_finish = clock() / static_cast<double>(CLOCKS_PER_SEC);
	show_time(calculation_start, calculation_finish, "time");

	common_finish = clock() / static_cast<double>(CLOCKS_PER_SEC);
	show_time(common_start, common_finish, "Common time");
	return 0;
}