cmake_minimum_required(VERSION 3.21)
set(PROJECT_NAME cufht)
project(${PROJECT_NAME})


#enable_language(CUDA)
option(USE_CUDA "Use cuda" ON)
find_package(CUDA REQUIRED)

add_exe(${PROJECT_NAME})

target_include_directories( ${PROJECT_NAME} PRIVATE ${RAPIDHT_INCLUDE_DIRS})
set_target_properties (${PROJECT_NAME} PROPERTIES FOLDER Demos)