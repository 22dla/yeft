cmake_minimum_required(VERSION 3.21)
set(PROJECT_NAME fhtCuda)
set(PROJECT_PATH ${CMAKE_CURRENT_SOURCE_DIR})
project(${PROJECT_NAME})

enable_language(CUDA)
find_package(CUDA REQUIRED)

add_exe(${PROJECT_NAME})