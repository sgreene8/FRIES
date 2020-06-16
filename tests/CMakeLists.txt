add_executable(unit_tests unit_tests.cpp)

target_link_libraries(unit_tests PRIVATE fries)

target_sources(unit_tests
  PRIVATE
    test_compression.cpp
    test_hamiltonian.cpp
    test_vector.cpp
)

add_test(NAME all COMMAND unit_tests --hf_path ${CMAKE_CURRENT_LIST_DIR}/../Input_Data/Neon_augccpvdz/ --out_path ${CMAKE_CURRENT_LIST_DIR})
## -s -n alias
