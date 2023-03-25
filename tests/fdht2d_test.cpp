#include <rapidht.h>
#include <utilities.h>

int main()
{
    // Define global 3D array sizes
    size_t cols = (size_t)pow(2, 9);
    size_t rows = cols;
    size_t image_num = 50;

    auto a3 = make_data_3d<float>(cols, rows, image_num);

    double common_start, common_finish;
    common_start = clock() / static_cast<double>(CLOCKS_PER_SEC);

    for (int i0 = 0; i0 < image_num; ++i0) {
        FDHT2D(&a3[i0]);
    }

    common_finish = clock() / static_cast<double>(CLOCKS_PER_SEC);
    show_time(common_start, common_finish, "Common time");
    return 0;
}