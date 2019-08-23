
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
        vec_utils.c)

add_library(fries "")
target_sources(fries PRIVATE ${FRIES_FILELIST})

add_subdirectory(Ext_Libs)
