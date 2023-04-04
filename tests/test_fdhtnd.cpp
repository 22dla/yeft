#include <rapidht.h>
#include <utilities.h>

int main()
{
    // Define global 3D array sizes
    int cols = (int)pow(2, 10);
    int rows = cols;
    int image_num = 10;

    auto a3 = make_data<double>({ cols, rows, image_num });

    double common_start, common_finish;
    common_start = clock() / static_cast<double>(CLOCKS_PER_SEC);

    auto ptr = a3.data();

    /*RapiDHT::HartleyTransform ht(cols, rows, image_num);
    for (int i0 = 0; i0 < image_num; ++i0) {
        ht.FDHT2D(ptr + i0 * rows * cols, rows, cols);
    }*/

    common_finish = clock() / static_cast<double>(CLOCKS_PER_SEC);
    show_time(common_start, common_finish, "Common time");
    return 0;
}