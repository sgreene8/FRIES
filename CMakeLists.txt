cmake_minimum_required(VERSION 3.6)

project(FRIES LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

if(NOT ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel"))
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wno-unused-result")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wno-unused-result")
endif()

if(NOT (("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang") OR ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")))
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -restrict -xavx -Wno-unknown-pragmas")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xavx -Wno-unknown-pragmas")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include(GNUInstallDirs)

add_subdirectory(FRIES)
add_subdirectory(FRIES_bin)
add_subdirectory(LAPACK)

message(STATUS "Project will be installed to ${CMAKE_INSTALL_PREFIX}")

# To uninstall, run "xargs rm < install_manifest.txt" from the build directory

enable_testing()

add_subdirectory(tests)
