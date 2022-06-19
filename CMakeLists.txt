cmake_minimum_required(VERSION 3.21)
project(fast-furious-transforms)
include(cmake/functions.cmake)

init_project(fast-furious-transforms)
# option(BUILD_FBX_READER "Build FBX Reader" OFF)
# add_definitions(-DCONFIG_PATH=\"${ENGINE_PATH}/projects/config_files_demo\")


add_subdirectory(projects/fht)
add_subdirectory(projects/fhtCuda)

