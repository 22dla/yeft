cmake_minimum_required(VERSION 3.21)
set(PROJECT_NAME cufht)
project(${PROJECT_NAME})

option(USE_CUDA "Use cuda" ON)
# Cuda
# Version 10.2 at least is needed for particular operations with memory management
find_package(CUDA 10.2 QUIET)
if(CUDA_FOUND)
	enable_language(CUDA)
	list(APPEND RAPIDHT_INCLUDE_DIRS ${CUDA_TOOLKIT_INCLUDE})
	if(${CUDA_VERSION} VERSION_LESS "10.0")
		find_library(CUDA_nvToolsExt_LIBRARY nvToolsExt)
		list(APPEND MINICERES_LIBRARIES ${CUDA_nvToolsExt_LIBRARY})
	endif()
endif(CUDA_FOUND)

add_exe(${PROJECT_NAME})

target_include_directories( ${PROJECT_NAME} PRIVATE ${RAPIDHT_INCLUDE_DIRS})
set_target_properties (${PROJECT_NAME} PROPERTIES FOLDER Demos)