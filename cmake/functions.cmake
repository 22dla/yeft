###########################
# Build project together
###########################

macro(init_path)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin) # exe
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib) # library
endmacro(init_path)

macro(init_openmp)
    option(USE_OPENMP "Use openmp" ON)
    if(USE_OPENMP)
        add_definitions(-DPE_USE_OMP)
        find_package(OpenMP)
        if(APPLE)
            execute_process(COMMAND brew --prefix libomp OUTPUT_VARIABLE BREW_LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
            set(OpenMP_omp_LIBRARY "${BREW_LIBOMP_PREFIX}/lib/libomp.dylib")
            set(OpenMP_INCLUDE_DIR "${BREW_LIBOMP_PREFIX}/include")
            message(STATUS "Using Homebrew libomp from ${BREW_LIBOMP_PREFIX}")
            include_directories("${OpenMP_INCLUDE_DIR}")
            # set (LIB_FILES ${OpenMP_omp_LIBRARY})
            list(APPEND PROJECT_LIB_FILES ${OpenMP_omp_LIBRARY})
        endif()
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endmacro(init_openmp)

###########################
# Build project separately
###########################

macro(init_project proj_name)
	option(USE_CUDA "Use cuda" ON)
    if(NOT_DEFINED_CMAKE_CUDA_ARCHITECTURE)
        set(CMAKE_CUDA_ARCHITECTURES 52 61 70 72 75 CACHE STRING "CUDA architectures" FORCE) 
    endif()
endmacro()

macro(add_exe exe_name)
	#project src files
	file(GLOB_RECURSE cpp_files ${PROJECT_PATH}/*.cpp)
	file(GLOB_RECURSE h_files ${PROJECT_PATH}/*.h)
    if(USE_CUDA)
		file(GLOB_RECURSE cu_files ${PROJECT_PATH}/*.cu)
		file(GLOB_RECURSE cuh_files ${PROJECT_PATH}/*.cuh)
	else()
		set(cu_files "")
        set(cuh_files "")
    endif(USE_CUDA)
    list(APPEND src_files ${h_files} ${cpp_files} ${cu_files} ${cuh_files})
    message(STATUS "add exe " ${exe_name} " with files " ${src_files})
    add_executable(${exe_name} ${src_files})
endmacro()

macro(add_lib lib_name src_path)
	#project src files
	file(GLOB_RECURSE cpp_files ${src_path}/*.cpp)
	file(GLOB_RECURSE h_files ${src_path}/*.h)
    if(USE_CUDA)
		file(GLOB_RECURSE cu_files ${src_path}/*.cu)
        file(GLOB_RECURSE cuh_files ${src_path}/*.cuh)
        message(STATUS "Use cuda in lib " ${lib_name})
	else()
		set(cu_files "")
        set(cuh_files "")
        set_target_properties(${lib_name} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    endif(USE_CUDA)
    list(APPEND src_files ${h_files} ${cpp_files} ${cu_files} ${cuh_files})
    message(STATUS "add lib " ${lib_name} " with files " ${src_files})
    add_library(${lib_name} ${src_files})
endmacro()
