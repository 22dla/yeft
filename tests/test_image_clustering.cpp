#include <iostream>
#include <rapidht.h>
#include <utilities.h>
#include <cmath>
#include <cstring>
#include <CommandLineParser.h>
#include "../3dparty/EasyBMP/EasyBMP.h"


int main(int argc, char* argv[]) {
	CommandLineParser parser(argc, argv);

	std::string inputFileName = parser.GetInputFileName();
	std::string outputFileName = parser.GetOutputFileName();

	if (inputFileName.empty() || outputFileName.empty()) {
		std::cerr << "Using: " << argv[0] << " -i input_image.bmp -o output_image.bmp" << std::endl;
		return 1;
	}

	// Чтение изображения
	BMP image;
	if (!image.ReadFromFile(inputFileName.c_str())) {
		std::cerr << "Could not read image." << std::endl;
		return 1;
	}

	int width = image.TellWidth();
	int height = image.TellHeight();

	std::vector<std::vector<uint8_t>> grayscaleData(height, std::vector<uint8_t>(width, 0));

	// Заполните вектор данными о серых оттенках из изображения
	/*for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			RGBApixel* pixel = image(x, y);
			uint8_t grayscaleValue = (uint8_t)((pixel->Red + pixel->Green + pixel->Blue) / 3);
			grayscaleData[y][x] = grayscaleValue;
		}
	}*/

	//auto ptr = pixelArray;
	//RapiDHT::HartleyTransform ht(height, width, 0, RapiDHT::CPU);
	//ht.ForwardTransform(ptr);

	if (image.WriteToFile(outputFileName.c_str())) {
		std::cout << "Image saved." << std::endl;
	}
	else {
		std::cerr << "Image is not saved." << std::endl;
	}

	return 0;
}
