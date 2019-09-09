
add_executable(fciqmc_mol fciqmc_mol.c)
target_link_libraries(fciqmc_mol PRIVATE fries)

add_executable(frimulti_mol frimulti_mol.c)
target_link_libraries(frimulti_mol PRIVATE fries)

add_executable(frisys_mol frisys_mol.c)
target_link_libraries(frisys_mol PRIVATE fries)

add_executable(fciqmc_hh fciqmc_hh.c)
target_link_libraries(fciqmc_hh PRIVATE fries)

add_executable(frisys_hh frisys_hh.c)
target_link_libraries(frisys_hh PRIVATE fries)
