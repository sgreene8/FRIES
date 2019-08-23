cmake_minimum_required(VERSION 3.14)

project(FRIes LANGUAGES C)

add_subdirectory(FRIes)
add_subdirectory(FRIes_bin)

target_link_libraries(fries PRIVATE dcmt)

option(USE_MPI "Enable MPI" ON)
if(USE_MPI)
    find_package(MPI REQUIRED)
    target_link_libraries(fries PRIVATE MPI::MPI_C)
    target_compile_definitions(fries PRIVATE USE_MPI)
endif()
