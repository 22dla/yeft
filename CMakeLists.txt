cmake_minimum_required(VERSION 3.21)
project(fast-furious-transforms)
include(cmake/functions.cmake)
use_openmp_local()

# option(BUILD_FBX_READER "Build FBX Reader" OFF)

set(LIBRARY_PATH ${CMAKE_CURRENT_SOURCE_DIR})

# add_definitions(-DCONFIG_PATH=\"${ENGINE_PATH}/projects/config_files_demo\")


add_subdirectory(${LIBRARY_PATH}/projects/fht)

