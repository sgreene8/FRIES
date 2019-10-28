add_executable(unit_tests unit_tests.cpp)

target_link_libraries(unit_tests PRIVATE fries)

target_sources(unit_tests
  PRIVATE
    test_compression.cpp
    test_hamiltonian.cpp
    catch.hpp
)

add_test(NAME compression COMMAND unit_tests --hf_path ${CMAKE_CURRENT_LIST_DIR}/../examples/Neon_augccpvdz_HF/ --out_path ${CMAKE_CURRENT_LIST_DIR} -s -n alias)
