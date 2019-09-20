cmake_minimum_required(VERSION 3.6)

project(FRIES LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include(GNUInstallDirs)

add_subdirectory(FRIES)
add_subdirectory(FRIES_bin)

message(STATUS "Project will be installed to ${CMAKE_INSTALL_PREFIX}")

install(
  TARGETS
    fciqmc_mol
    frimulti_mol
    frisys_mol
    fciqmc_hh
    frisys_hh
    fries
  RUNTIME
    DESTINATION ${INSTALL_BINDIR}
    COMPONENT bin
  LIBRARY
    DESTINATION ${INSTALL_LIBDIR}
    COMPONENT lib
  PUBLIC_HEADER
    DESTINATION ${INSTALL_INCLUDEDIR}/FRIES
    COMPONENT dev
  )

# To uninstall, run "xargs rm < install_manifest.txt" from the build directory

enable_testing()

add_subdirectory(tests)
