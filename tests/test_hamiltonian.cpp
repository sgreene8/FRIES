/*! \file
*
* \brief Tests for evaluating Hamiltonian matrix elements
*/

#include "catch.hpp"
#include <stdio.h>
#include <FRIES/Hamiltonians/molecule.h>
#include <FRIES/vec_utils.h>
#include <FRIES/io_utils.h>
#include <iostream>

TEST_CASE("Test diagonal matrix element evaluation", "[molec_diag]") {
    std::cout << "hello" << std::endl;
//    hf_input in_data;
//    parse_hf_input(hf_path, &in_data);
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
//    printf("My path is %s\n", argv[1]);
//    
//    long long hf_det = gen_hf_bitstring(n_orb, n_elec - n_frz);
//    
//    unsigned char tmp_orbs[n_elec_unf];
//    byte_table *table = gen_byte_table();
//    gen_orb_list(hf_det, table, tmp_orbs);
//    
//    double diag_el = diag_matrel(tmp_orbs, n_orb, eris, h_core, n_frz, n_elec);
//    
//    REQUIRE(diag_el == Approx(hf_en).margin(1e-8));
}
