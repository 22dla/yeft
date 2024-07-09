#include <iostream>
#include <vector>
#include <filesystem>
#include <utilities.h>

// Пространство имен для удобного использования файловой системы
namespace fs = std::filesystem;

int main() {
	// Путь к папке с изображениями
	std::string folderPath = "D:\\work\\images\\testImages";

    // Загрузка изображений и создание 3D массива
    std::vector<std::vector<std::vector<uint8_t>>> image3D = loadImagesTo3DArray(folderPath);

    // Проверка успешности загрузки изображений
    if (image3D.empty()) {
        std::cerr << "Loading is fail." << std::endl;
        return -1;
    }

	// Теперь у вас есть 3D массив изображений в переменной image3D
	std::cout << "Loaded " << image3D.size() << " images with sizes "
		<< image3D.front().size() << "x" << image3D.front().front().size() << std::endl;

	return 0;
}

