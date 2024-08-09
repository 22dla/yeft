#include <iostream>
#include <rapidht.h>
#include <utilities.h>
#include <cmath>
#include <numeric>
#include <cstring>

int main(int argc, char** argv) {

	size_t cols = static_cast<size_t>(pow(2, 14));
	size_t rows = static_cast<size_t>(pow(2, 14));
	RapiDHT::Modes mode = RapiDHT::CPU;

	// Обрабатываем аргументы командной строки, если они есть
	auto args_map = parseCommandLine(argc, argv);
	cols = parseSize(args_map, "--cols").value_or(cols);
	rows = parseSize(args_map, "--rows").value_or(rows);
	mode = parseMode(args_map).value_or(mode);

	// Выводим полученные значения
	std::cout << "Cols: " << cols << std::endl;
	std::cout << "Rows: " << rows << std::endl;
	std::cout << "Mode: " << (mode == RapiDHT::CPU ? "CPU" :
		mode == RapiDHT::GPU ? "GPU" : "RFFT") << std::endl << std::endl;

	auto a2_1 = makeData<double>({ cols, rows });
	auto a2_2(a2_1);

	double common_start, common_finish;
	common_start = clock() / static_cast<double>(CLOCKS_PER_SEC);

	//printData2D(a2_1.data(), rows, cols);

	RapiDHT::HartleyTransform ht(cols, rows, 0, mode);
	ht.ForwardTransform(a2_1);
	ht.InverseTransform(a2_1);

	//printData2D(a2_1.data(), rows, cols);

	common_finish = clock() / static_cast<double>(CLOCKS_PER_SEC);
	showTime(common_start, common_finish, "Common time");

	// Считаем ошибку
	double sum = std::transform_reduce(
		a2_1.begin(), a2_1.end(), a2_2.begin(), 0.0,
		std::plus<>(),
		[](double x, double y) { return std::abs(x - y); }
	);
	std::cout << "Error:\t" << sum << std::endl;
	return 0;
}
