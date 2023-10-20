add_library(EasyBMP STATIC
    EasyBMP/EasyBMP.cpp
    EasyBMP/EasyBMP.h
    EasyBMP/EasyBMP_BMP.h
    EasyBMP/EasyBMP_DataStructures.h
    EasyBMP/EasyBMP_VariousBMPutilities.h
)

target_include_directories(EasyBMP
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
# installing of library
install(TARGETS EasyBMP
        RUNTIME DESTINATION bin)