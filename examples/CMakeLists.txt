cmake_minimum_required(VERSION 3.6)
enable_testing()

find_package(fries REQUIRED)

add_executable(compress_1D compress_1D.c)
target_link_libraries(compress_1D fries)

add_test(NAME compress_1D COMMAND compress_1D)
