cmake_minimum_required(VERSION 3.21)
set(PROJECT_NAME fht)
project(${PROJECT_NAME})

find_package(OpenMP)
find_package(MPI)

IF(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ENDIF()

add_exe(${PROJECT_NAME})

message("${MPI_INCLUDE_PATH}")
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
target_include_directories( ${PROJECT_NAME} PRIVATE ${RAPIDHT_INCLUDE_DIRS})
set_target_properties (${PROJECT_NAME} PROPERTIES FOLDER Demos)

	