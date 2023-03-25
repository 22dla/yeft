# adding tests
add_executable(test_fdht1d fdht1d_test.cpp)
add_executable(test_fdht2d fdht2d_test.cpp)

# linking coreht to tests
target_link_libraries(test_fdht1d coreht)
target_link_libraries(test_fdht2d coreht)
