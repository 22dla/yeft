#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <Eigen/Dense>

// Пространство имен для удобного использования файловой системы
namespace fs = std::filesystem;

int main() {
    // Путь к папке с изображениями
    std::string folderPath = "D:\\work\\images\\testImages";

    // Вектор для хранения всех изображений
    std::vector<cv::Mat> images;

    // Обход всех файлов в указанной папке
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        // Проверка, что файл является изображением (опционально)
        if (entry.is_regular_file() && (entry.path().extension() == ".jpg" || entry.path().extension() == ".png" || entry.path().extension() == ".bmp")) {
            // Загрузка изображения
            cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);

            // Проверка успешности загрузки
            if (img.empty()) {
                std::cerr << "Ошибка загрузки изображения: " << entry.path().string() << std::endl;
                continue;
            }

            // Добавление изображения в вектор
            images.push_back(img);
        }
    }

    // Проверка, что изображения были загружены
    if (images.empty()) {
        std::cerr << "Нет изображений для загрузки." << std::endl;
        return -1;
    }

    // Создание 3D массива изображений
    // Размеры: количество изображений x высота x ширина x количество каналов
    int numImages = images.size();
    int height = images[0].rows;
    int width = images[0].cols;
    int channels = images[0].channels();

    // Создание 3D матрицы для хранения всех изображений
    cv::Mat image3D(numImages, height, width, CV_8UC1);

    for (int i = 0; i < numImages; ++i) {
        // Копирование данных изображения в 3D массив
        images[i].copyTo(image3D.row(i));
    }

    // Теперь у вас есть 3D массив изображений в переменной image3D
    std::cout << "Загружено " << numImages << " изображений размером " << height << "x" << width << std::endl;

    return 0;
}
