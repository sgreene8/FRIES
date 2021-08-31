
add_executable(fciqmc_mol fciqmc_mol.cpp)
target_link_libraries(fciqmc_mol PRIVATE fries)

add_executable(fciqmc_fp_mol fciqmc_fp_mol.cpp)
target_link_libraries(fciqmc_fp_mol PRIVATE fries)

add_executable(frimulti_mol frimulti_mol.cpp)
target_link_libraries(frimulti_mol PRIVATE fries)

add_executable(frisys_mol frisys_mol.cpp)
target_link_libraries(frisys_mol PRIVATE fries)

add_executable(frifull_hh frifull_hh.cpp)
target_link_libraries(frifull_hh PRIVATE fries)

add_executable(frifull_mol frifull_mol.cpp)
target_link_libraries(frifull_mol PRIVATE fries)

add_executable(frisys_hh frisys_hh.cpp)
target_link_libraries(frisys_hh PRIVATE fries)

add_executable(subspfull_mol subspfull_mol.cpp)
target_link_libraries(subspfull_mol PRIVATE fries)
target_link_libraries(subspfull_mol PRIVATE cnpy)
target_link_libraries(subspfull_mol PRIVATE my_lapack)

add_executable(subsp_mol subsp_mol.cpp)
target_link_libraries(subsp_mol PRIVATE fries)
target_link_libraries(subsp_mol PRIVATE cnpy)
target_link_libraries(subsp_mol PRIVATE my_lapack)

add_executable(subsp_mol_lowmem subsp_mol_lowmem.cpp)
target_link_libraries(subsp_mol_lowmem PRIVATE fries)
target_link_libraries(subsp_mol_lowmem PRIVATE cnpy)
target_link_libraries(subsp_mol_lowmem PRIVATE my_lapack)

add_executable(dice_dots dice_dots.cpp)
target_link_libraries(dice_dots PRIVATE fries)

add_executable(observables_mol observables_mol.cpp)
target_link_libraries(observables_mol PRIVATE fries)

add_executable(obs_repl_mol obs_repl_mol.cpp)
target_link_libraries(obs_repl_mol PRIVATE fries)

install(
    TARGETS
        fciqmc_mol
        fciqmc_fp_mol
        frimulti_mol
        frisys_mol
        frifull_hh
        frifull_mol
        subspfull_mol
        subsp_mol
        frisys_hh
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
