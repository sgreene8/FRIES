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

TEST_CASE("Test use of Matrix<bool>", "[bool]") {
    Matrix<bool> test_mat(2, 76);
    
    test_mat(0, 6) = true;
    test_mat(0, 70) = true;
    test_mat(1, 30) = true;
    
    uint64_t *row0 = test_mat[0];
    REQUIRE(row0[0] == uint64_t(0x1) << 6);
    REQUIRE(row0[1] == uint64_t(0x1) << (70 - 64));
    
    uint64_t *row1 = test_mat[1];
    REQUIRE(row1[0] == uint64_t(0x1) << 30);
}
