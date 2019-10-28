
add_executable(fciqmc_mol fciqmc_mol.cpp)
target_link_libraries(fciqmc_mol PRIVATE fries)

add_executable(frimulti_mol frimulti_mol.cpp)
target_link_libraries(frimulti_mol PRIVATE fries)

add_executable(frisys_mol frisys_mol.cpp)
target_link_libraries(frisys_mol PRIVATE fries)

add_executable(fciqmc_hh fciqmc_hh.cpp)
target_link_libraries(fciqmc_hh PRIVATE fries)

add_executable(frisys_hh frisys_hh.cpp)
target_link_libraries(frisys_hh PRIVATE fries)


install(
    TARGETS
        fciqmc_mol
        frimulti_mol
        frisys_mol
        fciqmc_hh
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
