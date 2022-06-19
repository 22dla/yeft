cmake_minimum_required(VERSION 3.21)
set(PROJECT_NAME fht)
set(PROJECT_PATH ${CMAKE_CURRENT_SOURCE_DIR})
project(${PROJECT_NAME})

find_package(OpenMP)
find_package(MPI)

IF(OPENMP_FOUND)
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ENDIF()


add_exe(${PROJECT_NAME})

include_directories(SYSTEM ${MPI_INCLUDE_PATH})
target_link_libraries(${PROJECT_NAME} PRIVATE ${MPI_C_LIBRARIES})
target_include_directories( ${PROJECT_NAME} PRIVATE ${PROJECT_PATH})