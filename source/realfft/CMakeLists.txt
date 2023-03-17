cmake_minimum_required(VERSION 3.21)
set(PROJECT_NAME realfft)
project(${PROJECT_NAME})

add_exe(${PROJECT_NAME})

target_include_directories( ${PROJECT_NAME} PRIVATE ${RAPIDHT_INCLUDE_DIRS})
set_target_properties (${PROJECT_NAME} PROPERTIES FOLDER Demos)