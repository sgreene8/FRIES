cmake_minimum_required(VERSION 3.6)
enable_testing()

add_executable(compress_1D compress_1D.c)
target_include_directories(compress_1D PRIVATE "${CMAKE_INSTALL_PREFIX}/include")

find_library(FRIES_LIB fries PATHS "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
target_link_libraries(compress_1D PRIVATE "${FRIES_LIB}")

find_library(DCMT_LIB dcmt  PATHS "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
target_link_libraries(compress_1D PRIVATE "${DCMT_LIB}")

add_test(NAME compress_1D COMMAND compress_1D)
