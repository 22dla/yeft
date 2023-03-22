#include <utilities.h>

void show_time(double startTime, double finishTime, std::string message)
{
    std::cout << message + ":\t" << finishTime - startTime << " sec" << std::endl;
}

template <typename T>
void write_data(const std::vector<T>& vec, int mode,
    const std::string& name, const std::string& path)
{
    std::fstream file;
    file.open(path, mode);asd

    switch (mode) {
    case std::ios_base::out: {

        file << ";";
        for (int i = 0; i < vec.size(); ++i) {
            file << i << ";";
        }

        file << std::endl
             << name << ";";
        for (int i = 0; i < vec.size(); ++i) {
            file << vec[i] << ";";
        }
        break;
    }
    case std::ios_base::app: {
        file << std::endl
             << name << ";";
        for (int i = 0; i < vec.size(); ++i) {
            file << vec[i] << ";";
        }
        break;
    }
    default:
        break;
    }

    file.close();
}

template <typename T>
std::vector<std::vector<std::vector<T>>> make_data_3d(int n, int m, int l)
{
    std::vector<std::vector<std::vector<T>>> data(l);

    for (size_t j1 = 0; j1 < l; ++j1) {
        data[j1].resize(n);
        for (size_t j2 = 0; j2 < n; ++j2) {
            data[j1][j2].resize(m);
            for (size_t j3 = 0; j3 < m; ++j3) {
                data[j1][j2][j3] = static_cast<T>(n + j1 + j2 + j3 + 2 + l) / (m);
            }
        }
    }
    return data;
}

// explicit template instantiation for int and float
template std::vector<std::vector<std::vector<int>>> make_data_3d(int n, int m, int l);
template std::vector<std::vector<std::vector<float>>> make_data_3d(int n, int m, int l);