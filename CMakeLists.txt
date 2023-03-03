cmake_minimum_required(VERSION 3.0)
project(RapidHT 
VERSION 0.1
)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)

# functions for internal use
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(cmake/functions.cmake)

set(RAPIDHT_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(RAPIDHT_INCLUDE_DIRS "${RAPIDHT_INCLUDE_DIRS}" CACHE STRING "RapidHT include dirs" FORCE)


# add projects
add_subdirectory(source)

