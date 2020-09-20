cmake_minimum_required(VERSION 3.6)

add_executable(fries_test fries_test.cpp)
target_include_directories(fries_test PRIVATE "${CMAKE_INSTALL_PREFIX}/include")

find_library(FRIES_LIB fries PATHS "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
target_link_libraries(fries_test PRIVATE "${FRIES_LIB}")
