cmake_minimum_required(VERSION 3.6)

project(FRIES_TEST LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(fries_test fries_test.cpp)
target_include_directories(fries_test PRIVATE "${CMAKE_INSTALL_PREFIX}/include")

find_library(FRIES_LIB fries PATHS "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
target_link_libraries(fries_test PRIVATE "${FRIES_LIB}")

find_package(MPI REQUIRED)
target_include_directories(fries_test PRIVATE ${MPI_INCLUDE_PATH})
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "CLANG")
    target_link_libraries(fries_test PRIVATE MPI::MPI_CXX)
else()
    target_link_libraries(fries_test PRIVATE ${MPI_LIBRARIES})
endif()
