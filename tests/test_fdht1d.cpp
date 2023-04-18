#include <rapidht.h>
#include <utilities.h>
#include <iostream>
#include <cmath>
#include <cstring>

int main(int argc, char** argv) {
	int rows = static_cast<int>(pow(2, 2));
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

	auto a1 = make_data<double>({ rows });
	auto a2 = make_data<double>({ rows });
	//print_data_1d(a1);

	auto ptr = a1.data();

	double common_start, common_finish;
	common_start = clock() / static_cast<double>(CLOCKS_PER_SEC);

	RapiDHT::HartleyTransform ht(rows, 0, 0, mode);
	
	ht.ForwardTransform(ptr);
	ht.InverseTransform(ptr);

	//print_data_1d(a1);

	common_finish = clock() / static_cast<double>(CLOCKS_PER_SEC);
	show_time(common_start, common_finish, "Common time");

	double sum = 0;
	for (int i = 0; i < rows; ++i) {
		sum += std::abs(a2[i] - ptr[i]);
		//std::cout << i << " " << a2[i] << " " << ptr[i] << " " << std::abs(a2[i] - ptr[i]) << std::endl;
	}
	std::cout << "Error:\t" << sum << std::endl;

	return 0;
}
