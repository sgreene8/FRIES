
list(APPEND FRIES_FILELIST
        compress_utils.c
        det_store.c
        fci_utils.c
        Hamiltonians/heat_bathPP.c
        Hamiltonians/hub_holstein.c
        Hamiltonians/near_uniform.c
        io_utils.c
        math_utils.c
        mpi_switch.h
        vec_utils.c
        Ext_Libs/argparse.c
        Ext_Libs/csvparser.c
        Ext_Libs/heap.c)

add_library(fries SHARED ${FRIES_FILELIST})

add_subdirectory(Ext_Libs)

target_link_libraries(fries PUBLIC dcmt)

option(USE_MPI "Enable MPI" ON)
if(USE_MPI)
    find_package(MPI REQUIRED)
#    target_link_libraries(fries PUBLIC ${MPI_LIBRARIES})
    target_link_libraries(fries PUBLIC MPI::MPI_C)
    target_compile_definitions(fries PUBLIC USE_MPI)
endif()
