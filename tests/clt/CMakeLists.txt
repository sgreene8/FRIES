cmake_minimum_required(VERSION 3.6)

set(CMAKE_CXX_STANDARD 14)

add_executable(clt main.cpp)

target_include_directories(clt PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../../")

find_library(FRIES_LIB fries PATHS "${CMAKE_CURRENT_LIST_DIR}/../../build/FRIES")

target_link_libraries(clt PRIVATE "${FRIES_LIB}")

target_compile_definitions(clt PUBLIC USE_MPI)

find_package(MPI REQUIRED)
target_include_directories(clt PUBLIC ${MPI_INCLUDE_PATH})
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "CLANG")
    target_link_libraries(clt PUBLIC MPI::MPI_CXX)
else()
    target_link_libraries(clt PUBLIC ${MPI_LIBRARIES})
endif()
