
list(APPEND FRIES_FILELIST
        compress_utils.cpp
        det_store.c
        fci_utils.c
        Hamiltonians/heat_bathPP.cpp
        Hamiltonians/hub_holstein.c
        Hamiltonians/near_uniform.cpp
        Hamiltonians/molecule.cpp
        io_utils.cpp
        math_utils.c
        vec_utils.cpp
        Ext_Libs/argparse.c
        Ext_Libs/csvparser.c
        Ext_Libs/heap.c)

add_library(fries SHARED ${FRIES_FILELIST})

set_target_properties(fries
  PROPERTIES
PUBLIC_HEADER "compress_utils.hpp;det_store.h;fci_utils.h;io_utils.hpp;math_utils.h;mpi_switch.h;vec_utils.hpp")

add_subdirectory(Ext_Libs)
add_subdirectory(Hamiltonians)

target_link_libraries(fries PUBLIC dcmt)
target_link_libraries(fries PUBLIC m)

option(USE_MPI "Enable MPI" ON)
if(USE_MPI)
    find_package(MPI REQUIRED)
#    target_link_libraries(fries PUBLIC ${MPI_LIBRARIES})
#    target_link_libraries(fries PUBLIC MPI::MPI_C)
    target_link_libraries(fries PUBLIC MPI::MPI_CXX)
    target_compile_definitions(fries PUBLIC USE_MPI)
endif()



install(
    TARGETS
        fries
    RUNTIME
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}
        COMPONENT bin
    LIBRARY
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
        COMPONENT lib
    PUBLIC_HEADER
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/FRIES
        COMPONENT dev
)
