cmake_minimum_required(VERSION 3.6)

project(FRIES LANGUAGES C)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(FRIES)
add_subdirectory(FRIES_bin)

set(CMAKECONFIG_INSTALL_DIR "~/bin/fries")

install(TARGETS fciqmc_mol frimulti_mol frisys_mol fciqmc_hh frisys_hh DESTINATION ${CMAKECONFIG_INSTALL_DIR})
