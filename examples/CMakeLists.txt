cmake_minimum_required(VERSION 3.6)
enable_testing()

add_executable(compress_1D compress_1D.c)
target_include_directories(compress_1D PRIVATE "${SEARCH_DIRECTORY}/include/")

#find_library(FRIES_LIB fries PATHS "${SEARCH_DIRECTORY}/lib/")
#message(STATUS "Result is ${FRIES_LIB}")
#target_link_libraries(compress_1D PRIVATE "${FRIES_LIB}")

add_library(fries SHARED IMPORTED)
set_target_properties(fries PROPERTIES IMPORTED_LOCATION /usr/local/lib/libfries.dylib)
target_link_libraries(compress_1D PRIVATE fries)

add_test(NAME compress_1D COMMAND compress_1D)
