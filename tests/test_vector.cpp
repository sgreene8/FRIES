/*! \file
*
* \brief Tests related to indexing and storing elements in a vector
*/

#include "catch.hpp"
#include <string.h>
#include <FRIES/vec_utils.hpp>
#include <FRIES/Hamiltonians/hub_holstein.hpp>

TEST_CASE("Test bit string math", "[binary]") {
    uint8_t bit_str[3];
    bit_str[0] = 0b11000000;
    bit_str[1] = 0b00111000;
    bit_str[2] = 0b010;
    
    REQUIRE(bits_between(bit_str, 1, 7) == 1);
    REQUIRE(bits_between(bit_str, 14, 2) == 5);
    REQUIRE(bits_between(bit_str, 1, 3) == 0);
    REQUIRE(bits_between(bit_str, 18, 12) == 2);
    
    uint8_t bit_str2[3];
    memcpy(bit_str2, bit_str, 3);
    
    REQUIRE(bit_str_equ(bit_str, bit_str2, 3) == 1);
    REQUIRE(read_bit(bit_str, 11) == 1);
    REQUIRE(read_bit(bit_str, 10) == 0);
    
    zero_bit(bit_str2, 11);
    REQUIRE(read_bit(bit_str2, 11) == 0);
    set_bit(bit_str2, 10);
    REQUIRE(read_bit(bit_str2, 10) == 1);
    
    REQUIRE(bit_str_equ(bit_str, bit_str2, 3) == 0);
}


TEST_CASE("Test vector indexing with bit strings", "[vec_idx]") {
    uint8_t bit_str1[6];
    uint8_t bit_str2[6];
    bit_str2[0] = 255;
    bit_str2[1] = 249;
    bit_str2[2] = 15;
    
    size_t n_bytes = 3;
    unsigned int n_orb = 11;
    unsigned int n_elec = 18;
    
    gen_hf_bitstring(n_orb, n_elec, bit_str1);
    REQUIRE(bit_str_equ(bit_str1, bit_str2, n_bytes) == 1);
    
    n_elec = 20;
    n_orb = 18;
    n_bytes = 5;
    bit_str2[1] = 3;
    bit_str2[2] = 252;
    bit_str2[3] = 15;
    bit_str2[4] = 0;
    gen_hf_bitstring(n_orb, n_elec, bit_str1);
    REQUIRE(bit_str_equ(bit_str1, bit_str2, n_bytes) == 1);
    
    n_elec = 10;
    n_orb = 24;
    n_bytes = 6;
    bit_str2[0] = 31;
    bit_str2[1] = 0;
    bit_str2[2] = 0;
    bit_str2[3] = 31;
    bit_str2[4] = 0;
    bit_str2[5] = 0;
    gen_hf_bitstring(n_orb, n_elec, bit_str1);
    REQUIRE(bit_str_equ(bit_str1, bit_str2, n_bytes) == 1);
}
