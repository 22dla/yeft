cmake_minimum_required(VERSION 3.21)
project(fast-furious-transforms)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)

# functions for internal use
include(cmake/functions.cmake)

# add projects
add_subdirectory(projects/fht)
add_subdirectory(projects/fhtCuda)
#add_subdirectory(projects/tests)
