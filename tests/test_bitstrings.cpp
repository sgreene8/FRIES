/*! \file
 *
 * \brief Tests related to manipulation of bit strings as representations of Slater determinants
 */

#include "catch.hpp"
#include <cstring>
#include <FRIES/det_store.h>
#include <FRIES/fci_utils.h>
#include <FRIES/Hamiltonians/hub_holstein.hpp>
#include <cstdint>

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
    
    REQUIRE((!memcmp(bit_str, bit_str2, 3)) == 1);
    REQUIRE(read_bit(bit_str, 11) == 1);
    REQUIRE(read_bit(bit_str, 10) == 0);
    
    zero_bit(bit_str2, 11);
    REQUIRE(read_bit(bit_str2, 11) == 0);
    set_bit(bit_str2, 10);
    REQUIRE(read_bit(bit_str2, 10) == 1);
    
    REQUIRE((!memcmp(bit_str, bit_str2, 3)) == 0);
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
    
    std::fill(bit_str1, bit_str1 + n_bytes, 255);
    gen_hf_bitstring(n_orb, n_elec, bit_str1);
    REQUIRE((!memcmp(bit_str1, bit_str2, n_bytes)) == 1);
    
    n_elec = 20;
    n_orb = 18;
    n_bytes = 5;
    bit_str2[1] = 3;
    bit_str2[2] = 252;
    bit_str2[3] = 15;
    bit_str2[4] = 0;
    std::fill(bit_str1, bit_str1 + n_bytes, 255);
    gen_hf_bitstring(n_orb, n_elec, bit_str1);
    REQUIRE((!memcmp(bit_str1, bit_str2, n_bytes)) == 1);
    
    n_elec = 10;
    n_orb = 24;
    n_bytes = 6;
    bit_str2[0] = 31;
    bit_str2[1] = 0;
    bit_str2[2] = 0;
    bit_str2[3] = 31;
    bit_str2[4] = 0;
    bit_str2[5] = 0;
    std::fill(bit_str1, bit_str1 + n_bytes, 255);
    gen_hf_bitstring(n_orb, n_elec, bit_str1);
    REQUIRE((!memcmp(bit_str1, bit_str2, n_bytes)) == 1);
    
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
    std::fill(bit_str1, bit_str1 + n_bytes, 255);
    gen_hf_bitstring(n_orb, n_elec, bit_str1);
    REQUIRE((!memcmp(bit_str1, bit_str2, n_bytes)) == 1);
}


TEST_CASE("Test generation of Neel bit strings", "[neel_bits]") {
    unsigned int n_elec = 6;
    unsigned int n_sites = 6;
    
    uint8_t str1[10];
    uint8_t str2[10];
    
    gen_neel_det_1D(n_sites, n_elec, 0, str1);
    str2[0] = 0b10010101;
    str2[1] = 0b1010;
    
    REQUIRE(!memcmp(str1, str2, 2));
    
    n_elec = 2;
    n_sites = 2;
    gen_neel_det_1D(n_sites, n_elec, 0, str1);
    str2[0] = 0b1001;
    
    REQUIRE(!memcmp(str1, str2, 1));
    
    n_elec = 8;
    n_sites = 8;
    gen_neel_det_1D(n_sites, n_elec, 0, str1);
    REQUIRE(str1[0] == 0b01010101);
    REQUIRE(str1[1] == 0b10101010);
    
    n_elec = 4;
    n_sites = 4;
    gen_neel_det_1D(n_sites, n_elec, 0, str1);
    
    str2[0] = 0b10100101;
    REQUIRE(!memcmp(str1, str2, 1));
    
    n_sites = 10;
    n_elec = 10;
    uint8_t ph_bits = 3;
    gen_neel_det_1D(n_sites, n_elec, ph_bits, str1);
    REQUIRE(str1[0] == 0b01010101);
    REQUIRE(str1[1] == 0b10101001);
    REQUIRE(str1[2] == 0b1010);
    REQUIRE(str1[3] == 0);
    REQUIRE(str1[4] == 0);
    REQUIRE(str1[5] == 0);
    REQUIRE(str1[6] == 0);
}

TEST_CASE("Test determinant spin-flipping", "[flip_spins]") {
    uint8_t str1[6];
    uint8_t str2[6];
    
    str1[0] = 0b11000000;
    str1[1] = 0b00000011;
    str1[2] = 0b1111;
    flip_spins(str1, str2, 10);
    REQUIRE(!memcmp(str1, str2, 3));
    
    str1[0] = 0b11001100;
    str1[1] = 0b00110011;
    flip_spins(str1, str2, 8);
    str1[0] = 0b00110011;
    str1[1] = 0b11001100;
    REQUIRE(!memcmp(str1, str2, 2));
    
    flip_spins(str1, str2, 8);
    str1[0] = 0b11001100;
    str1[1] = 0b00110011;
    REQUIRE(!memcmp(str1, str2, 2));
    
    str1[0] = 0b11000011;
    flip_spins(str1, str2, 4);
    REQUIRE(str2[0] == 0b00111100);
    
    str1[0] = 0b11110001;
    str1[1] = 0b11100001;
    str1[2] = 0b111;
    flip_spins(str1, str2, 12);
    str1[0] = 0b01111110;
    str1[1] = 0b00010000;
    str1[2] = 0b00011111;
    REQUIRE(!memcmp(str1, str2, 3));
    
    str1[0] = 0b00001011;
    str1[1] = 0b1110;
    flip_spins(str1, str2, 6);
    str1[0] = 0b11111000;
    str1[1] = 0b0010;
    REQUIRE(!memcmp(str1, str2, 2));
    
    str1[0] = 0b01000100;
    str1[1] = 0b11000000;
    str1[2] = 0b00111000;
    str1[3] = 0b1;
    flip_spins(str1, str2, 16);
    str1[0] = 0b00111000;
    str1[1] = 0b1;
    str1[2] = 0b01000100;
    str1[3] = 0b11000000;
    REQUIRE(!memcmp(str1, str2, 4));
    
    str1[0] = 0b1100;
    str1[1] = 0b100001;
    str1[2] = 0b11000000;
    str1[3] = 0b11;
    str1[4] = 0;
    str1[5] = 0;
    flip_spins(str1, str2, 22);
    str1[0] = 0b1111;
    str1[1] = 0;
    str1[2] = 0;
    str1[3] = 0b1000011;
    str1[4] = 0b1000;
    str1[5] = 0;
    REQUIRE(!memcmp(str1, str2, 6));
    
    str1[0] = 0b00001100;
    str1[1] = 0b00100000;
    str1[2] = 0b11000001;
    str1[3] = 0b00000011;
    str1[4] = 0;
    str1[5] = 0;
    flip_spins(str1, str2, 22);
    str1[0] = 0b00001111;
    str1[1] = 0;
    str1[2] = 0;
    str1[3] = 0b00000011;
    str1[4] = 0b01001000;
    str1[5] = 0;
    REQUIRE(!memcmp(str1, str2, 6));
}


TEST_CASE("Test identification of excitations", "[id_excite]") {
    uint8_t n_orb = 10;
    uint8_t n_elec = 8;
    uint8_t n_bytes = CEILING(2 * n_orb, 8);
    uint8_t det1[n_bytes];
    uint8_t det2[n_bytes];
    uint8_t orbs1[5] = {0, 0, 0, 0, 0};
    
    gen_hf_bitstring(n_orb, n_elec, det1);
    std::copy(det1, det1 + n_bytes, det2);
    
    uint8_t bit_diff = find_excitation(det1, det2, orbs1, n_bytes);
    REQUIRE(bit_diff == 0);
    
    orbs1[0] = 2;
    orbs1[1] = 9;
    sing_det_parity(det2, orbs1);
    
    bit_diff = find_excitation(det1, det2, orbs1, n_bytes);
    REQUIRE(bit_diff == 1);
    REQUIRE(orbs1[0] == 2);
    REQUIRE(orbs1[1] == 9);
    REQUIRE(orbs1[2] == 0);
    REQUIRE(orbs1[3] == 0);
    REQUIRE(orbs1[4] == 0);
    
    orbs1[0] = 0;
    orbs1[1] = 7;
    sing_det_parity(det2, orbs1);
    bit_diff = find_excitation(det1, det2, orbs1, n_bytes);
    REQUIRE(bit_diff == 2);
    REQUIRE(orbs1[0] == 0);
    REQUIRE(orbs1[1] == 2);
    REQUIRE(orbs1[2] == 7);
    REQUIRE(orbs1[3] == 9);
    REQUIRE(orbs1[4] == 0);
    
    orbs1[0] = 11;
    orbs1[1] = 20;
    sing_det_parity(det2, orbs1);
    bit_diff = find_excitation(det1, det2, orbs1, n_bytes);
    REQUIRE(bit_diff == UINT8_MAX);
    
    orbs1[0] = 7;
    orbs1[1] = 2;
    sing_det_parity(det2, orbs1);
    bit_diff = find_excitation(det1, det2, orbs1, n_bytes);
    REQUIRE(bit_diff == 2);
    REQUIRE(orbs1[0] == 0);
    REQUIRE(orbs1[1] == 11);
    REQUIRE(orbs1[2] == 9);
    REQUIRE(orbs1[3] == 20);
    REQUIRE(orbs1[4] == 0);
}

TEST_CASE("Test spin-flip connections", "[flip_connect]") {
    uint32_t n_elec = 8;
    uint32_t n_orb = 10;
    uint8_t orbs[10];
    uint8_t idx[2];
    
    orbs[0] = 1;
    orbs[1] = 2;
    orbs[2] = 3;
    orbs[3] = 4;
    orbs[4] = 11;
    orbs[5] = 12;
    orbs[6] = 13;
    orbs[7] = 14;
    REQUIRE(tr_doub_connect(orbs, n_orb, n_elec, idx) == 0);
    
    orbs[0] = 1;
    orbs[1] = 2;
    orbs[2] = 3;
    orbs[3] = 4;
    orbs[4] = 11;
    orbs[5] = 12;
    orbs[6] = 13;
    orbs[7] = 15;
    REQUIRE(tr_doub_connect(orbs, n_orb, n_elec, idx) == 1);
    REQUIRE(idx[0] == 3);
    REQUIRE(idx[1] == 7);
    
    orbs[0] = 1;
    orbs[1] = 2;
    orbs[2] = 3;
    orbs[3] = 4;
    orbs[4] = 11;
    orbs[5] = 12;
    orbs[6] = 14;
    orbs[7] = 15;
    REQUIRE(tr_doub_connect(orbs, n_orb, n_elec, idx) == 1);
    REQUIRE(idx[0] == 2);
    REQUIRE(idx[1] == 7);
    
    orbs[0] = 1;
    orbs[1] = 2;
    orbs[2] = 4;
    orbs[3] = 5;
    orbs[4] = 11;
    orbs[5] = 12;
    orbs[6] = 13;
    orbs[7] = 14;
    REQUIRE(tr_doub_connect(orbs, n_orb, n_elec, idx) == 1);
    REQUIRE(idx[0] == 3);
    REQUIRE(idx[1] == 6);
    
    orbs[0] = 1;
    orbs[1] = 2;
    orbs[2] = 4;
    orbs[3] = 5;
    orbs[4] = 11;
    orbs[5] = 12;
    orbs[6] = 15;
    orbs[7] = 16;
    REQUIRE(tr_doub_connect(orbs, n_orb, n_elec, idx) == 1);
    REQUIRE(idx[0] == 2);
    REQUIRE(idx[1] == 7);
    
    orbs[0] = 1;
    orbs[1] = 2;
    orbs[2] = 3;
    orbs[3] = 4;
    orbs[4] = 11;
    orbs[5] = 12;
    orbs[6] = 15;
    orbs[7] = 16;
    REQUIRE(tr_doub_connect(orbs, n_orb, n_elec, idx) == 2);
    
    orbs[0] = 1;
    orbs[1] = 3;
    orbs[2] = 5;
    orbs[3] = 7;
    orbs[4] = 13;
    orbs[5] = 15;
    orbs[6] = 16;
    orbs[7] = 17;
    REQUIRE(tr_doub_connect(orbs, n_orb, n_elec, idx) == 1);
    REQUIRE(idx[0] == 0);
    REQUIRE(idx[1] == 6);
}
