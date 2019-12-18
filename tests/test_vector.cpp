/*! \file
*
* \brief Tests related to indexing and storing elements in a vector
*/

#include "catch.hpp"
#include <string.h>
#include <FRIES/hh_vec.hpp>
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


TEST_CASE("Test generation of Hartree-Fock bit strings", "[hf_bits]") {
    uint8_t bit_str1[7];
    uint8_t bit_str2[7];
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
    
    n_elec = 10;
    n_orb = 26;
    n_bytes = 7;
    bit_str2[0] = 31;
    bit_str2[1] = 0;
    bit_str2[2] = 0;
    bit_str2[3] = 124;
    bit_str2[4] = 0;
    bit_str2[5] = 0;
    bit_str2[6] = 0;
    gen_hf_bitstring(n_orb, n_elec, bit_str1);
    REQUIRE(bit_str_equ(bit_str1, bit_str2, n_bytes) == 1);
}


TEST_CASE("Test encoding and decoding of Holstein basis states", "[phonon_bits]") {
    uint8_t n_sites = 5;
    uint8_t ph_bits = 3;
    uint8_t det_size = 4;
    
    uint8_t det1[det_size];
    uint8_t ph_nums1[] = {5, 2, 0, 7, 1};
    uint8_t ph_nums2[n_sites];
    
    det1[1] = 0b01010100;
    det1[2] = 0b01111000;
    det1[3] = 0;
    
    mt_struct *rngen_ptr = get_mt_parameter_id_st(32, 521, 0, (unsigned int) time(NULL));
    sgenrand_mt((uint32_t) time(NULL), rngen_ptr);
    HubHolVec<int> sol_vec(1, 0, rngen_ptr, n_sites, ph_bits, 0, 1);
    
    sol_vec.decode_phonons(det1, ph_nums2);
    
    for (size_t site_idx = 0; site_idx < n_sites; site_idx++) {
        REQUIRE(ph_nums2[site_idx] == ph_nums1[site_idx]);
    }
    
    uint8_t excite_det[det_size];
    sol_vec.det_from_ph(det1, excite_det, 2, 1);
    ph_nums1[2]++;
    sol_vec.decode_phonons(excite_det, ph_nums2);
    
    for (size_t site_idx = 0; site_idx < n_sites; site_idx++) {
        REQUIRE(ph_nums2[site_idx] == ph_nums1[site_idx]);
    }
}
