#include <rapidht.h>
#include <utilities.h>

int main() {
    // Define global 3D array sizes
    size_t cols = (size_t)pow(2, 10);
    size_t rows = cols;
    size_t image_num = 10;

    auto a3 = make_data_3d<float>(cols, rows, image_num);

    double common_start, common_finish;
    common_start = clock() / static_cast<double>(CLOCKS_PER_SEC);

    auto ptr = a3.data();

    
    for (int i0 = 0; i0 < image_num; ++i0) {
        RapiDHT::HartleyTransform ht(cols, rows);
        ht.ForwardTransform(ptr + i0 * rows * cols);
    }

    common_finish = clock() / static_cast<double>(CLOCKS_PER_SEC);
    show_time(common_start, common_finish, "Common time");
    return 0;
}