
add_executable(fciqmc fciqmc.c)
target_link_libraries(fciqmc PRIVATE fries)

add_executable(fri fri.c)
target_link_libraries(fri PRIVATE fries)

add_executable(fri_sys fri_sys.c)
target_link_libraries(fri_sys PRIVATE fries)

add_executable(hh_qmc hh_qmc.c)
target_link_libraries(hh_qmc PRIVATE fries)

#add_executable(hh_sys hh_sys.c)
#target_link_libraries(hh_sys PRIVATE fries)
