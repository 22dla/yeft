cmake_minimum_required(VERSION 3.18)

project(fht)

enable_language(CUDA)
find_package(CUDAToolkit)

add_exe(fht)

use_freeglut_local(${PROJECT_NAME})
use_cuda_local(${PROJECT_NAME})

		