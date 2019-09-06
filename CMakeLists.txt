cmake_minimum_required(VERSION 3.6)

project(FRIES LANGUAGES C)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(FRIES)
add_subdirectory(FRIES_bin)
