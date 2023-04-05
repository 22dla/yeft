# Find CUDA
find_package(CUDA REQUIRED)
enable_language(CUDA)
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_61")

# Find OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# adding files from source
file(GLOB SOURCES *.cpp *.cu)
add_library(coreht ${SOURCES})

include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})
target_link_libraries(coreht PRIVATE ${CUDA_LIBRARIES})
#set_target_properties(coreht PROPERTIES CUDA_ARCHITECTURES "60;61")

# installing of library
install(TARGETS coreht
        RUNTIME DESTINATION bin)


# it helped!
#export CPATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/11.8/include/:$CPATH
#export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/11.8/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH
#export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/11.8/bin/:$PATH
