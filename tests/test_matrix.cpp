/*! \file
 * \brief Testing the functionality of the Matrix class
 */

#include "catch.hpp"
#include <FRIES/ndarr.hpp>
#include <cstdio>

TEST_CASE("Test reshaping of a matrix", "[reshape]") {
    Matrix<uint8_t> test_mat(4, 4);
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            test_mat(i, j) = i * 4 + j;
        }
    }
    
    int keep[] = {4, 4, 4, 4};
    test_mat.enlarge_cols(6, keep);
    
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 4; j < 6; j++) {
            test_mat(i, j) = 100;
        }
    }
    
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            REQUIRE(test_mat(i, j) == i * 4 + j);
        }
    }
}
