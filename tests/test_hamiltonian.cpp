/*! \file
*
* \brief Tests for evaluating Hamiltonian matrix elements
*/

#include "catch.hpp"
#include "inputs.hpp"
#include <stdio.h>
#include <FRIES/Hamiltonians/molecule.hpp>
#include <FRIES/Hamiltonians/hub_holstein.hpp>
#include <FRIES/hh_vec.hpp>
#include <FRIES/io_utils.hpp>

TEST_CASE("Test diagonal matrix element evaluation", "[molec_diag]") {
    using namespace test_inputs;
    hf_input in_data;
    parse_hf_input(hf_path.c_str(), &in_data);
    unsigned int n_elec = in_data.n_elec;
    unsigned int n_frz = in_data.n_frz;
    unsigned int n_orb = in_data.n_orb;
    unsigned int tot_orb = n_orb + n_frz / 2;
    unsigned int n_elec_unf = n_elec - n_frz;
    double hf_en = in_data.hf_en;

    Matrix<double> *h_core = in_data.hcore;
    FourDArr *eris = in_data.eris;
    
    // Construct DistVec object
    // Rn generator
    mt_struct *rngen_ptr = get_mt_parameter_id_st(32, 521, 0, (unsigned int) time(NULL));
    sgenrand_mt((uint32_t) time(NULL), rngen_ptr);
    DistVec<double> sol_vec(10, 0, rngen_ptr, 2 * n_orb, n_elec_unf, 1);
    
    uint8_t *hf_det = sol_vec.indices()[0];
    gen_hf_bitstring(n_orb, n_elec - n_frz, hf_det);

    uint8_t tmp_orbs[n_elec_unf];
    REQUIRE(sol_vec.gen_orb_list(hf_det, tmp_orbs) == n_elec_unf);

    double diag_el = diag_matrel(tmp_orbs, tot_orb, *eris, *h_core, n_frz, n_elec);

    REQUIRE(diag_el == Approx(hf_en).margin(1e-8));
}

TEST_CASE("Test enumeration of symmetry-allowed single excitations", "[single_symm]") {
    uint8_t symm[] = {0, 1, 2, 0, 1, 1};
    uint8_t test_det[2];
    test_det[0] = 0b10001110;
    test_det[1] = 0b1100;
    uint8_t occ_orbs[] = {1, 2, 3, 7, 10, 11};
    uint8_t excitations[5][2];
    uint8_t answer[][2] = {{1, 4}, {1, 5}, {3, 0}};
    
    size_t n_sing = sing_ex_symm(test_det, occ_orbs, 6, 6, excitations, symm);
    REQUIRE(n_sing == 3);
    
    size_t idx1, idx2;
    for (idx1 = 0; idx1 < n_sing; idx1++) {
        for (idx2 = 0; idx2 < 2; idx2++) {
            REQUIRE((int)excitations[idx1][idx2] == (int)answer[idx1][idx2]);
        }
    }
}


TEST_CASE("Test evaluation of Hubbard matrix elements", "[hubbard]") {
    unsigned int n_sites = 10;
    size_t n_bytes = CEILING(2 * n_sites, 8);
    uint8_t det[n_bytes];
    byte_table *tabl = gen_byte_table();
    
    // 1010000101 0011000011
    det[0] = 0b11000011;
    det[1] = 0b00010100;
    det[2] = 0b1010;
    REQUIRE(hub_diag(det, n_sites, tabl) == 2);
    
    n_sites = 8;
    // 10001111 11000111
    det[0] = 0b11000111;
    det[1] = 0b10001111;
    REQUIRE(hub_diag(det, n_sites, tabl) == 4);
    
    n_sites = 3;
    // 101 011
    det[0] = 0b11101011;
    REQUIRE(hub_diag(det, n_sites, tabl) == 1);
    
    n_sites = 4;
    det[0] = 0b00100010;
    REQUIRE(hub_diag(det, n_sites, tabl) == 1);
    
    n_sites = 5;
    det[0] = 0b1000010;
    det[1] = 0;
    REQUIRE(hub_diag(det, n_sites, tabl) == 1);
    
    n_sites = 5;
    det[0] = 255;
    det[1] = 255;
    REQUIRE(hub_diag(det, n_sites, tabl) == n_sites);
}


TEST_CASE("Test calculation of overlap with neel state in Hubbard model", "[neel_ovlp]") {
    uint8_t bit_str1[5];
    uint8_t bit_str2[5];
    
    unsigned int n_sites = 4;
    unsigned int n_elec = 4;
    
    gen_neel_det_1D(n_sites, n_elec, bit_str1);
    bit_str2[0] = 165;
    bit_str2[1] = 0;
    
    // Returning correct neel state
    REQUIRE(bit_str_equ(bit_str1, bit_str2, 2));
    
    byte_table *tabl = gen_byte_table();
    // Diagonal element for neel state should be zero
    REQUIRE(hub_diag(bit_str1, n_sites, tabl) == 0);
    
    n_sites = 20;
    n_elec = 8;
    gen_neel_det_1D(n_sites, n_elec, bit_str1);
    bit_str2[0] = 85;
    bit_str2[1] = 0;
    bit_str2[2] = 160;
    bit_str2[3] = 10;
    bit_str2[4] = 0;
    // Returning correct neel state
    REQUIRE(bit_str_equ(bit_str1, bit_str2, 5));
    
    // Diagonal element for neel state should be zero
    REQUIRE(hub_diag(bit_str1, n_sites, tabl) == 0);
    
    n_sites = 10;
    n_elec = 10;
    gen_neel_det_1D(n_sites, n_elec, bit_str1);
    bit_str2[0] = 0b01010101;
    bit_str2[1] = 0b10101001;
    bit_str2[2] = 0b1010;
    
    // Returning correct neel state
    REQUIRE(bit_str_equ(bit_str1, bit_str2, 3));
    
    // Diagonal element for neel state should be zero
    REQUIRE(hub_diag(bit_str1, n_sites, tabl) == 0);
    
    // correctly ignore bits after 2 * n_sites
    bit_str2[2] = 0b11111010;
    REQUIRE(hub_diag(bit_str2, n_sites, tabl) == 0);
    
    zero_bit(bit_str2, 15);
    set_bit(bit_str2, 16);
    
    Matrix<uint8_t> dets(2, 4);
    memcpy(dets[0], bit_str2, 4);
    memcpy(dets[1], bit_str2, 4);
    zero_bit(dets[1], 8);
    set_bit(dets[1], 9);
    
    int vals[] = {2, 3};
    // Correctly identify excitation across left byte boundary
    REQUIRE(calc_ref_ovlp(dets, vals, 2, bit_str1, tabl, n_elec, n_sites) == 2);
    
    zero_bit(dets[0], 16);
    set_bit(dets[0], 15);
    set_bit(dets[0], 7);
    zero_bit(dets[0], 8);
    
    // Correctly identify excitation across right byte boundary
    REQUIRE(calc_ref_ovlp(dets, vals, 2, bit_str1, tabl, n_elec, n_sites) == 2);
    
    // correctly ignore bits after 2 * n_sites
    set_bit(dets[0], 8);
    zero_bit(dets[0], 7);
    dets[0][2] = 250;
    REQUIRE(calc_ref_ovlp(dets, vals, 2, bit_str1, tabl, n_elec, n_sites) == 0);
}


TEST_CASE("Test generation of excitations in the Hubbard model", "[hub_excite]") {
    uint8_t det[4];
    uint8_t n_sites = 10;
    uint8_t n_elec = 10;
    
    gen_neel_det_1D(n_sites, n_elec, det);
    
    uint8_t correct_orbs[][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {11, 12}, {13, 14}, {15, 16}, {17, 18}, {2, 1}, {4, 3}, {6, 5}, {8, 7}, {11, 10}, {13, 12}, {15, 14}, {17, 16}, {19, 18}};
    
    uint8_t test_orbs[18][2];
    
    mt_struct *rngen_ptr = get_mt_parameter_id_st(32, 521, 0, (unsigned int) time(NULL));
    sgenrand_mt((uint32_t) time(NULL), rngen_ptr);
    
    // Solution vector
    HubHolVec<int> sol_vec(1, 0, rngen_ptr, n_sites, 0, n_elec, 1);
    
    Matrix<uint8_t> &neighb = sol_vec.neighb();
    sol_vec.find_neighbors_1D(det, neighb[0]);
    
    REQUIRE(hub_all(n_elec, neighb[0], test_orbs) == 18);
    
    size_t ex_idx;
    for (ex_idx = 0; ex_idx < 18; ex_idx++) {
        REQUIRE(correct_orbs[ex_idx][0] == test_orbs[ex_idx][0]);
        REQUIRE(correct_orbs[ex_idx][1] == test_orbs[ex_idx][1]);
    }
}

TEST_CASE("Test identification of empty neighboring orbitals in a Hubbard determinant", "[hub_neigh]") {
    uint8_t det[2];
    det[0] = 0b10010101;
    det[1] = 0b110;
    
    int n_elec = 6;
    int n_sites = 6;
    
    // Solution vector
    mt_struct *rngen_ptr = get_mt_parameter_id_st(32, 521, 0, (unsigned int) time(NULL));
    sgenrand_mt((uint32_t) time(NULL), rngen_ptr);
    HubHolVec<int> sol_vec(1, 0, rngen_ptr, n_sites, 0, n_elec, 1);
    
    uint8_t *neighb = sol_vec.neighb()[0];
    sol_vec.find_neighbors_1D(det, neighb);
    
    uint8_t real_neigb[] = {5, 0, 2, 4, 7, 10, 0, 4, 2, 4, 7, 9, 0, 0};
    uint8_t neigb_idx;
    
    REQUIRE(neighb[0] == real_neigb[0]);
    for (neigb_idx = 0; neigb_idx < neighb[0]; neigb_idx++) {
        REQUIRE(real_neigb[neigb_idx + 1] == neighb[neigb_idx + 1]);
    }
    REQUIRE(neighb[n_elec + 1] == real_neigb[n_elec + 1]);
    for (neigb_idx = 0; neigb_idx < neighb[n_elec + 1]; neigb_idx++) {
        REQUIRE(real_neigb[n_elec + 1 + neigb_idx + 1] == neighb[n_elec + 1 + neigb_idx + 1]);
    }
}


TEST_CASE("Test counting of singly/doubly occupied sites in a Hubbard-Holstein basis state", "[hub_sites]") {
    uint8_t det[2];
    det[0] = 0b01010101;
    det[1] = 0b110;
    
    int n_elec = 6;
    int n_sites = 6;
    
    // Solution vector
    mt_struct *rngen_ptr = get_mt_parameter_id_st(32, 521, 0, (unsigned int) time(NULL));
    sgenrand_mt((uint32_t) time(NULL), rngen_ptr);
    HubHolVec<int> sol_vec(1, 0, rngen_ptr, n_sites, 0, n_elec, 1);
    
    uint8_t occ[n_elec];
    sol_vec.gen_orb_list(det, occ);
    
    REQUIRE(idx_of_doub(0, n_elec, occ, det, n_sites) == 0);
    REQUIRE(idx_of_doub(1, n_elec, occ, det, n_sites) == 4);
    REQUIRE(idx_of_sing(0, n_elec, occ, det, n_sites) == 2);
    REQUIRE(idx_of_sing(1, n_elec, occ, det, n_sites) == 9);
}
