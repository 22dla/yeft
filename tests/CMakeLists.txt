# adding tests
add_executable(test1 test1.cpp)
add_executable(test2 test2.cpp)

# linking coreht to tests
target_link_libraries(test1 coreht)
target_link_libraries(test2 coreht)
