# Find CUDA
find_package(CUDA REQUIRED)
enable_language(CUDA)
# Find OpenMP
find_package(OpenMP REQUIRED)
# Find MPI
#find_package(MPI REQUIRED)
#include_directories(SYSTEM ${MPI_INCLUDE_PATH})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# adding files from source
file(GLOB SOURCES *.cpp *.cu)
add_library(coreht ${SOURCES})

include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})
#set_target_properties(coreht PROPERTIES CUDA_ARCHITECTURES "sm_60")
target_link_libraries(coreht PRIVATE ${CUDA_LIBRARIES})
#target_link_libraries(coreht PRIVATE ${MPI_C_LIBRARIES} ${CUDA_LIBRARIES})

# installing of library
install(TARGETS coreht
        RUNTIME DESTINATION bin)
