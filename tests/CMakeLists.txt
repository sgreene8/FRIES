add_executable(unit_tests unit_tests.cpp)

target_link_libraries(unit_tests PRIVATE fries)

target_sources(unit_tests
  PRIVATE
    test_compression.cpp
    test_hamiltonian.cpp
    test_vector.cpp
)

set(NUMPROC 2)
if (DEFINED ENV{NUM_MPI})
    set(NUMPROC $ENV{NUM_MPI})
endif()

if(USE_MPI)
    add_test(NAME all COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NUMPROC} unit_tests --hf_path ${CMAKE_CURRENT_LIST_DIR}/../Input_Data/Neon_augccpvdz/ --out_path ${CMAKE_CURRENT_LIST_DIR})
else()
    add_test(NAME all COMMAND unit_tests --hf_path ${CMAKE_CURRENT_LIST_DIR}/../Input_Data/Neon_augccpvdz/ --out_path ${CMAKE_CURRENT_LIST_DIR})
endif()
