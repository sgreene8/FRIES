/*! \file
*
* \brief Tests for evaluating Hamiltonian matrix elements
*/

#include "catch.hpp"
#include "inputs.hpp"
#include <stdio.h>
#include <FRIES/Hamiltonians/molecule.hpp>
#include <FRIES/Hamiltonians/hub_holstein.hpp>
#include <FRIES/Hamiltonians/heat_bathPP.hpp>
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
    std::vector<uint32_t> tmp;
    DistVec<double> sol_vec(10, 0, rngen_ptr, 2 * n_orb, n_elec_unf, 1, tmp);
    
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

TEST_CASE("Test counting of virtual orbitals in a determinant", "[virt_count]") {
    uint8_t n_elec = 6;
    uint8_t n_orb = 10;
    
    uint8_t occ_orbs[10];
    occ_orbs[0] = 1;
    occ_orbs[1] = 4;
    occ_orbs[2] = 7;
    occ_orbs[3] = 11;
    occ_orbs[4] = 15;
    occ_orbs[5] = 17;
    
    REQUIRE(find_nth_virt(occ_orbs, 0, n_elec, n_orb, 0) == 0);
    REQUIRE(find_nth_virt(occ_orbs, 0, n_elec, n_orb, 5) == 8);
    REQUIRE(find_nth_virt(occ_orbs, 1, n_elec, n_orb, 0) == 10);
    REQUIRE(find_nth_virt(occ_orbs, 1, n_elec, n_orb, 2) == 13);
    
    n_elec = 8;
    n_orb = 22;
    occ_orbs[0] = 0;
    occ_orbs[1] = 1;
    occ_orbs[2] = 2;
    occ_orbs[3] = 3;
    occ_orbs[4] = 22;
    occ_orbs[5] = 23;
    occ_orbs[6] = 24;
    occ_orbs[7] = 25;
    
    REQUIRE(find_nth_virt(occ_orbs, 1, n_elec, n_orb, 0) == 26);
    
    n_elec = 6;
    n_orb = 10;
    uint8_t symm[] = {0,1,2,0,1,2,0,1,2,0};
    Matrix<uint8_t> symm_lookup(n_irreps, n_orb + 1);
    gen_symm_lookup(symm, symm_lookup);
    
    uint8_t det[3];
    det[0] = 0b100100;
    det[1] = 0b111010;
    det[2] = 0;
    REQUIRE(find_nth_virt_symm(det, 0, 2, 0, symm_lookup) == 8);
    REQUIRE(find_nth_virt_symm(det, n_orb, 2, 1, symm_lookup) == 8 + n_orb);
    REQUIRE(find_nth_virt_symm(det, n_orb, 2, 2, symm_lookup) == 255);
}


TEST_CASE("Test evaluation of Hubbard matrix elements", "[hubbard]") {
    unsigned int n_sites = 10;
    size_t n_bytes = CEILING(2 * n_sites, 8);
    uint8_t det[n_bytes];
    
    // 1010000101 0011000011
    det[0] = 0b11000011;
    det[1] = 0b00010100;
    det[2] = 0b1010;
    REQUIRE(hub_diag(det, n_sites) == 2);
    
    n_sites = 8;
    // 10001111 11000111
    det[0] = 0b11000111;
    det[1] = 0b10001111;
    REQUIRE(hub_diag(det, n_sites) == 4);
    
    n_sites = 3;
    // 101 011
    det[0] = 0b11101011;
    REQUIRE(hub_diag(det, n_sites) == 1);
    
    n_sites = 4;
    det[0] = 0b00100010;
    REQUIRE(hub_diag(det, n_sites) == 1);
    
    n_sites = 5;
    det[0] = 0b1000010;
    det[1] = 0;
    REQUIRE(hub_diag(det, n_sites) == 1);
    
    n_sites = 5;
    det[0] = 255;
    det[1] = 255;
    REQUIRE(hub_diag(det, n_sites) == n_sites);
}


TEST_CASE("Test calculation of overlap with neel state in Hubbard-Holstein", "[neel_ovlp]") {
    uint8_t bit_str1[10];
    uint8_t bit_str2[10];
    
    unsigned int n_sites = 4;
    unsigned int n_elec = 4;
    uint8_t ph_bits = 2;
    uint8_t det_size = CEILING((2 + ph_bits) * n_sites, 8);
    
    gen_neel_det_1D(n_sites, n_elec, ph_bits, bit_str1);
    bit_str2[0] = 165;
    bit_str2[1] = 0;
    
    // Returning correct neel state
    REQUIRE(bit_str_equ(bit_str1, bit_str2, det_size));
    
    // Diagonal element for neel state should be zero
    REQUIRE(hub_diag(bit_str1, n_sites) == 0);
    
    n_sites = 20;
    n_elec = 8;
    det_size = CEILING((2 + ph_bits) * n_sites, 8);
    
    gen_neel_det_1D(n_sites, n_elec, ph_bits, bit_str1);
    bit_str2[0] = 85;
    bit_str2[1] = 0;
    bit_str2[2] = 160;
    bit_str2[3] = 10;
    bit_str2[4] = 0;
    bzero(&bit_str2[5], 5);
    // Returning correct neel state
    REQUIRE(bit_str_equ(bit_str1, bit_str2, det_size));
    
    // Diagonal element for neel state should be zero
    REQUIRE(hub_diag(bit_str1, n_sites) == 0);
    
    n_sites = 10;
    n_elec = 10;
    det_size = CEILING((2 + ph_bits) * n_sites, 8);
    
    gen_neel_det_1D(n_sites, n_elec, ph_bits, bit_str1);
    bit_str2[0] = 0b01010101;
    bit_str2[1] = 0b10101001;
    bit_str2[2] = 0b1010;
    bit_str2[3] = 0;
    bit_str2[4] = 0;
    
    // Returning correct neel state
    REQUIRE(bit_str_equ(bit_str1, bit_str2, det_size));
    
    // Diagonal element for neel state should be zero
    REQUIRE(hub_diag(bit_str1, n_sites) == 0);
    
    // correctly ignore bits after 2 * n_sites
    bit_str2[2] = 0b11111010;
    REQUIRE(hub_diag(bit_str2, n_sites) == 0);
    bit_str2[2] = 0b1010;
    
    zero_bit(bit_str2, 15);
    set_bit(bit_str2, 16);
    
    Matrix<uint8_t> dets(2, det_size);
    memcpy(dets[0], bit_str2, det_size);
    memcpy(dets[1], bit_str2, det_size);
    zero_bit(dets[1], 8);
    set_bit(dets[1], 9);
    
    Matrix<uint8_t> phonons(2, n_sites);
    bzero(phonons[0], 2 * n_sites * sizeof(uint8_t));
    uint8_t occ1[] = {0, 2, 4, 6, 8, 11, 13, 15, 17, 19};
    
    int vals[] = {2, 3};
    // Correctly identify excitation across left byte boundary
    REQUIRE(calc_ref_ovlp(dets, vals, phonons, 2, bit_str1, occ1, n_elec, n_sites, 0) == 2);
    
    zero_bit(dets[0], 16);
    set_bit(dets[0], 15);
    set_bit(dets[0], 7);
    zero_bit(dets[0], 8);
    
    // Correctly identify excitation across right byte boundary
    REQUIRE(calc_ref_ovlp(dets, vals, phonons, 2, bit_str1, occ1, n_elec, n_sites, 0) == 2);
    
    // Correctly identify a phonon excitation
    set_bit(dets[0], 8);
    zero_bit(dets[0], 7);
    REQUIRE(bit_str_equ(bit_str1, dets[0], det_size));
    set_bit(dets[0], 2 * n_sites);
    phonons(0, 0) = 1;
    REQUIRE(calc_ref_ovlp(dets, vals, phonons, 2, bit_str1, occ1, n_elec, n_sites, 10) == -20);
    
    // Correctly ignore wrong phonon excitations
    phonons(0, 0) = 2;
    zero_bit(dets[0], 2 * n_sites);
    set_bit(dets[0], 2 * n_sites + 1);
    REQUIRE(calc_ref_ovlp(dets, vals, phonons, 2, bit_str1, occ1, n_elec, n_sites, 10) == 0);
}


TEST_CASE("Test generation of excitations in the Hubbard model", "[hub_excite]") {
    uint8_t det[4];
    uint8_t n_sites = 10;
    uint8_t n_elec = 10;
    
    gen_neel_det_1D(n_sites, n_elec, 0, det);
    
    uint8_t correct_orbs[][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {11, 12}, {13, 14}, {15, 16}, {17, 18}, {2, 1}, {4, 3}, {6, 5}, {8, 7}, {11, 10}, {13, 12}, {15, 14}, {17, 16}, {19, 18}};
    
    uint8_t test_orbs[18][2];
    
    mt_struct *rngen_ptr = get_mt_parameter_id_st(32, 521, 0, (unsigned int) time(NULL));
    sgenrand_mt((uint32_t) time(NULL), rngen_ptr);
    
    // Solution vector
    std::vector<uint32_t> tmp;
    HubHolVec<int> sol_vec(1, 0, rngen_ptr, n_sites, 0, n_elec, 1, NULL, 1, tmp);
    
    Matrix<uint8_t> &neighb = sol_vec.neighb();
    sol_vec.find_neighbors_1D(det, neighb[0]);
    
    REQUIRE(hub_all(n_elec, neighb[0], test_orbs) == 18);
    
    size_t ex_idx;
    for (ex_idx = 0; ex_idx < 18; ex_idx++) {
        REQUIRE(correct_orbs[ex_idx][0] == test_orbs[ex_idx][0]);
        REQUIRE(correct_orbs[ex_idx][1] == test_orbs[ex_idx][1]);
    }
    
    n_sites = 2;
    n_elec = 2;
    gen_neel_det_1D(n_sites, n_elec, 0, det);
    
    REQUIRE(det[0] == 0b1001);
    
    HubHolVec<int>sol_vec2(1, 0, rngen_ptr, n_sites, 1, n_elec, 1, NULL, 0, tmp);
    sol_vec2.find_neighbors_1D(det, neighb[0]);
    
    REQUIRE(neighb[0][0] == 1);
    REQUIRE(neighb[0][1] == 0);
    REQUIRE(neighb[0][n_elec + 1] == 1);
    REQUIRE(neighb[0][n_elec + 2] == 3);
    
    det[0] = 0b11001;
    sol_vec2.find_neighbors_1D(det, neighb[0]);
    REQUIRE(neighb[0][0] == 1);
    REQUIRE(neighb[0][1] == 0);
    REQUIRE(neighb[0][n_elec + 1] == 1);
    REQUIRE(neighb[0][n_elec + 2] == 3);
    
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
    std::vector<uint32_t> tmp;
    HubHolVec<int> sol_vec(1, 0, rngen_ptr, n_sites, 0, n_elec, 1, NULL, 1, tmp);
    
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
    std::vector<uint32_t> tmp;
    HubHolVec<int> sol_vec(1, 0, rngen_ptr, n_sites, 0, n_elec, 1, NULL, 1, tmp);
    
    uint8_t occ[n_elec];
    sol_vec.gen_orb_list(det, occ);
    
    REQUIRE(idx_of_doub(0, n_elec, occ, det, n_sites) == 0);
    REQUIRE(idx_of_doub(1, n_elec, occ, det, n_sites) == 4);
    REQUIRE(idx_of_sing(0, n_elec, occ, det, n_sites) == 2);
    REQUIRE(idx_of_sing(1, n_elec, occ, det, n_sites) == 9);
}


TEST_CASE("Test generation of phonon excitations/de-excitations in the Holstein model", "[hol_ex]") {
    uint8_t n_sites = 3;
    uint8_t n_elec = 4;
    uint8_t ph_bits = 2;
    size_t det_size = CEILING(2 * n_sites + ph_bits * n_sites, 8);
    
    mt_struct *rngen_ptr = get_mt_parameter_id_st(32, 521, 0, (unsigned int) time(NULL));
    sgenrand_mt((uint32_t) time(NULL), rngen_ptr);
    std::vector<uint32_t> tmp;
    HubHolVec<double> sol_vec(1, 0, rngen_ptr, n_sites, ph_bits, n_elec, 1, NULL, 1, tmp);
    
    uint8_t orig_det[det_size];
    orig_det[0] = 0b00101110;
    orig_det[1] = 0;
    
    uint8_t new_det[det_size];
    uint8_t correct_det[det_size];
    
    REQUIRE(sol_vec.det_from_ph(orig_det, new_det, 0, 1) == 1);
    correct_det[0] = 0b01101110;
    correct_det[1] = 0;
    REQUIRE(bit_str_equ(new_det, correct_det, det_size));
    
    REQUIRE(sol_vec.det_from_ph(orig_det, new_det, 1, 1) == 1);
    correct_det[0] = 0b00101110;
    correct_det[1] = 1;
    REQUIRE(bit_str_equ(new_det, correct_det, det_size));
    
    REQUIRE(sol_vec.det_from_ph(new_det, orig_det, 1, -1) == 1);
    correct_det[1] = 0;
    REQUIRE(bit_str_equ(orig_det, correct_det, det_size));
    
    REQUIRE(sol_vec.det_from_ph(orig_det, new_det, 1, -1) == 0);
    
    orig_det[1] = 3;
    REQUIRE(sol_vec.det_from_ph(orig_det, new_det, 1, 1) == 0);
}


TEST_CASE("Test evaluation of sampling weights for the new heat-bath distribution", "[new_hb]") {
    mt_struct *rngen_ptr = get_mt_parameter_id_st(32, 521, 0, (unsigned int) time(NULL));
    sgenrand_mt((uint32_t) time(NULL), rngen_ptr);
    unsigned int n_orb = 5;
    unsigned int n_elec = 4;
    size_t i, j, a, b;
    
    FourDArr eris(n_orb, n_orb, n_orb, n_orb);
    
    for (i = 0; i < n_orb; i++) {
        for (j = 0; j < n_orb; j++) {
            for (a = 0; a < n_orb; a++) {
                for (b = 0; b < n_orb; b++) {
                    eris(i, j, a, b) = genrand_mt(rngen_ptr) / (1. + UINT32_MAX) - 0.5;
                }
            }
        }
    }
    
    hb_info *hbtens = set_up(n_orb, n_orb, eris);
    
    uint8_t occ[] = {0, 2, 7, 8};
    double probs[n_orb * 2];
    bzero(probs, sizeof(double) * 2 * n_orb);
    
    double o1_norm = calc_o1_probs(hbtens, probs, n_elec, occ, 1);
    double norm_chk = 0;
    for (i = 1; i < n_elec; i++) {
        REQUIRE(probs[i - 1] * o1_norm == Approx(hbtens->s_tens[occ[i] % n_orb] / hbtens->s_norm).margin(1e-7));
        norm_chk += hbtens->s_tens[occ[i] % n_orb];
    }
    norm_chk /= hbtens->s_norm;
    REQUIRE(norm_chk == Approx(o1_norm).margin(1e-7));
    
    uint8_t o1 = 3;
    double o2_norm = calc_o2_probs_half(hbtens, probs, n_elec, occ, o1);
    o1 = occ[3];
    norm_chk = 0;
    for (j = 0; j < n_elec / 2; j++) {
        REQUIRE(probs[j] * o2_norm == Approx(hbtens->d_diff[(occ[3] - n_orb) * n_orb + occ[j]] / hbtens->s_tens[occ[3] % n_orb]).margin(1e-7));
        norm_chk += hbtens->d_diff[(occ[3] - n_orb) * n_orb + occ[j]];
    }
    
#define TRI_N(n)((n) * (n + 1) / 2)
#define I_J_TO_TRI(i, j)(TRI_N(j - 1) + i)
    REQUIRE(probs[2] * o2_norm == Approx(hbtens->d_same[I_J_TO_TRI(occ[2] - n_orb, occ[3] - n_orb)] / hbtens->s_tens[occ[3] % n_orb]).margin(1e-7));
    
    norm_chk += hbtens->d_same[I_J_TO_TRI(occ[2] - n_orb, occ[3] - n_orb)];
    norm_chk /= hbtens->s_tens[occ[3] % n_orb];
    REQUIRE(norm_chk == Approx(o2_norm).margin(1e-7));
    
    uint8_t o2 = occ[2];
    uint8_t u1 = 6;
    uint8_t u2 = 5;
    uint8_t orbs[] = {o2, o1, u2, u1};
    o1 %= n_orb;
    o2 %= n_orb;
    u1 %= n_orb;
    u2 %= n_orb;
    uint8_t min_o1_u1 = o1 < u1 ? o1 : u1;
    uint8_t max_o1_u1 = o1 > u1 ? o1 : u1;
    uint8_t min_o2_u2 = o2 < u2 ? o2 : u2;
    uint8_t max_o2_u2 = o2 > u2 ? o2 : u2;
    
    double weight = calc_unnorm_wt(hbtens, orbs);
    REQUIRE(weight == Approx(hbtens->d_same[I_J_TO_TRI(o2, o1)] / hbtens->s_norm * (hbtens->exch_sqrt[I_J_TO_TRI(min_o1_u1, max_o1_u1)] * hbtens->exch_sqrt[I_J_TO_TRI(min_o2_u2, max_o2_u2)]) / hbtens->exch_norms[o1] / hbtens->exch_norms[o2]).margin(1e-7));
}
