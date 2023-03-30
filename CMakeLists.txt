cmake_minimum_required(VERSION 3.0)
#cmake_policy(SET CMP0104 NEW)

# Name of project
project(RapiDHT)

# setting support of C++11
set(CMAKE_CXX_STANDARD 11)

# including includes and adding projects
include_directories(include)
add_subdirectory(source)
add_subdirectory(tests)

# setting tests
enable_testing()
add_test(NAME test1 COMMAND test1)
add_test(NAME test2 COMMAND test2)
