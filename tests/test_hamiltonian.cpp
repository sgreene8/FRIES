/*! \file
*
* \brief Tests for evaluating Hamiltonian matrix elements
*/

#include "catch.hpp"
#include "inputs.hpp"
#include <stdio.h>
#include <FRIES/Hamiltonians/molecule.hpp>
#include <FRIES/vec_utils.hpp>
#include <FRIES/io_utils.hpp>

TEST_CASE("Test diagonal matrix element evaluation", "[molec_diag]") {
//    using namespace test_inputs;
//    hf_input in_data;
//    parse_hf_input(hf_path.c_str(), &in_data);
//    unsigned int n_elec = in_data.n_elec;
//    unsigned int n_frz = in_data.n_frz;
//    unsigned int n_orb = in_data.n_orb;
//    unsigned int n_elec_unf = n_elec - n_frz;
//    unsigned int tot_orb = n_orb + n_frz / 2;
//    double hf_en = in_data.hf_en;
//    
//    double (* h_core)[tot_orb] = (double (*)[tot_orb])in_data.hcore;
//    double (* eris)[tot_orb][tot_orb][tot_orb] = (double (*)[tot_orb][tot_orb][tot_orb])in_data.eris;
//    
//    long long hf_det = gen_hf_bitstring(n_orb, n_elec - n_frz);
//    
//    unsigned char tmp_orbs[n_elec_unf];
//    byte_table *table = gen_byte_table();
//    gen_orb_list(hf_det, table, tmp_orbs);
//    
//    double diag_el = diag_matrel(tmp_orbs, n_orb, eris, hcore, n_frz, n_elec);
//    
//    REQUIRE(diag_el == Approx(hf_en).margin(1e-8));
}

//TEST_CASE("Test enumeration of symmetry-allowed single excitations", "[single_symm]") {
//    unsigned char symm[] = {0, 1, 2, 0, 1, 1};
//    long long test_det = 0b110010001110;
//    unsigned char occ_orbs[] = {1, 2, 3, 7, 10, 11};
//    unsigned char excitations[5][2];
//    unsigned char answer[][2] = {{1, 4}, {1, 5}, {3, 0}};
//    
//    size_t n_sing = sing_ex_symm(test_det, occ_orbs, 6, 6, excitations, symm);
//    REQUIRE(n_sing == 3);
//    
//    size_t idx1, idx2;
//    for (idx1 = 0; idx1 < n_sing; idx1++) {
//        for (idx2 = 0; idx2 < 2; idx2++) {
//            REQUIRE((int)excitations[idx1][idx2] == (int)answer[idx1][idx2]);
//        }
//    }
//}
