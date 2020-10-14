/*! \file
 * \brief Tests of functions defined in lapack_wrappers.cpp
*/

#include "catch.hpp"
#include <LAPACK/lapack_wrappers.hpp>
#include <FRIES/ndarr.hpp>
#include <algorithm>

TEST_CASE("Test calculation of singular values", "[s_vals]") {
    Matrix<double> mat(5, 5);
    double data[5][5] = {
        {0.74351344, 0.65469139, 0.42781707, 0.02688775, 0.26114885},
        {0.73301815, 0.60986924, 0.18794226, 0.89048779, 0.74340984},
        {0.23726301, 0.24448013, 0.57073834, 0.4524871 , 0.30715852},
        {0.41971749, 0.42878294, 0.45522689, 0.5917804 , 0.95092287},
        {0.55441452, 0.83238152, 0.43487077, 0.90679147, 0.67324495}};
    for (uint8_t idx = 0; idx < 5; idx++) {
        std::copy(data[idx], data[idx] + 5, mat[idx]);
    }
    
    std::vector<double> s_vals(5);
    double scratch[3 * 5 + 5 * 5 - 1 + 32 * (5 + 5)];
    get_svals(mat, s_vals, scratch);
    
    double ref_svals[] = {2.7705791 , 0.69476036, 0.47513809, 0.36429323, 0.18385449};
    for (uint8_t idx = 0; idx < 5; idx++) {
        REQUIRE(s_vals[idx] == Approx(ref_svals[idx]).margin(1e-5));
    }
}


TEST_CASE("Test calculation of generalized eigenvectors/values", "[ge_vecs_vals]") {
    Matrix<double> op_mat(5, 5);
    double data1[5][5] = {
        {0.74351344, 0.65469139, 0.42781707, 0.02688775, 0.26114885},
        {0.73301815, 0.60986924, 0.18794226, 0.89048779, 0.74340984},
        {0.23726301, 0.24448013, 0.57073834, 0.4524871 , 0.30715852},
        {0.41971749, 0.42878294, 0.45522689, 0.5917804 , 0.95092287},
        {0.55441452, 0.83238152, 0.43487077, 0.90679147, 0.67324495}};
    for (uint8_t idx = 0; idx < 5; idx++) {
        std::copy(data1[idx], data1[idx] + 5, op_mat[idx]);
    }

    Matrix<double> ovlp_mat(5, 5);
    double data2[5][5] = {
        {0.46096063, 0.37905362, 0.38972974, 0.7521    , 0.53220609},
        {0.77209488, 0.55504151, 0.99756786, 0.76508879, 0.41130084},
        {0.80482064, 0.60049142, 0.26426873, 0.89648412, 0.19994793},
        {0.03151442, 0.34905052, 0.50890564, 0.32582598, 0.55857986},
        {0.04447881, 0.74425435, 0.25162458, 0.64690051, 0.36743151}
    };
    for (uint8_t idx = 0; idx < 5; idx++) {
        std::copy(data2[idx], data2[idx] + 5, ovlp_mat[idx]);
    }

    std::vector<double> evals(5);
    Matrix<double> evecs(5, 5);
    double scratch[42 * 5];
    get_real_gevals_vecs(op_mat, ovlp_mat, evals, evecs);

    // sort by eigenvalue
    std::vector<uint8_t> sort_idx(5);
    for (uint8_t idx = 0; idx < 5; idx++) {
        sort_idx[idx] = idx;
    }
    auto comparator = [evals](uint8_t idx1, uint8_t idx2) {
        return evals[idx1] < evals[idx2];
    };
    std::sort(sort_idx.begin(), sort_idx.end(), comparator);
    
    // normalize by 2-norm for comparison
    for (uint8_t idx = 0; idx < 5; idx++) {
        double two_norm = 0;
        for (uint8_t idx2 = 0; idx2 < 5; idx2++) {
            two_norm += evecs(idx, idx2) * evecs(idx, idx2);
        }
        two_norm = sqrt(two_norm);
        for (uint8_t idx2 = 0; idx2 < 5; idx2++) {
            evecs(idx, idx2) /= two_norm;
        }
    }

    double ref_evals[] = {-6.23564511, -1.26152671,  0.29670827,  0.94192697,  1.49513739};
    double ref_evecs[5][5] = {
        {-0.41886477,  0.29651271, -0.64622294, -0.45971664, -0.02173683},
        {-0.52328305, -0.27337554,  0.67638159,  0.84429419, -0.480224  },
        { 0.18374559, -0.77206924, -0.13528511,  0.19387283, -0.06014759},
        { 0.69225751,  0.22504461, -0.2150935 ,  0.12410711,  0.35774055},
        {-0.19427882,  0.4365891 ,  0.24563507,  0.15112473, -0.79832098}
    };
    
    // flip eigenvector signs if necessary
    for (uint8_t eval_idx = 0; eval_idx < 5; eval_idx++) {
        if ((evecs(sort_idx[eval_idx], 0) > 0) ^ (ref_evecs[0][eval_idx] > 0)) {
            for (uint8_t el_idx = 0; el_idx < 5; el_idx++) {
                evecs(sort_idx[eval_idx], el_idx) *= -1;
            }
        }
    }

    for (uint8_t idx = 0; idx < 5; idx++) {
        REQUIRE(evals[sort_idx[idx]] == Approx(ref_evals[idx]).margin(1e-5));
    }

    for (uint8_t row_idx = 0; row_idx < 5; row_idx++) {
        for (uint8_t col_idx = 0; col_idx < 5; col_idx++) {
            REQUIRE(evecs(sort_idx[col_idx], row_idx) == Approx(ref_evecs[row_idx][col_idx]).margin(1e-5));
        }
    }
}

TEST_CASE("Test matrix inversion", "[mat_inv]") {
    Matrix<double> mat(2, 2);
    mat(0, 0) = 1;
    mat(1, 0) = 2;
    mat(0, 1) = 3;
    mat(1, 1) = 4;
    
    double scratch[4];
    inv_inplace(mat);
    
    double ref_inv[2][2] = {
        {-2, 1.5},
        {1, -0.5}
    };
    
    REQUIRE(mat(0, 0) == Approx(ref_inv[0][ 0]).margin(1e-7));
    REQUIRE(mat(0, 1) == Approx(ref_inv[0][ 1]).margin(1e-7));
    REQUIRE(mat(1, 0) == Approx(ref_inv[1][ 0]).margin(1e-7));
    REQUIRE(mat(1, 1) == Approx(ref_inv[1][ 1]).margin(1e-7));
}


TEST_CASE("Test QR factorization", "[qr]") {
    double input[] = {0, 3, 6, 1, 4, 7, 2, 5, 9};
    Matrix<double> mat(3, 3);
    std::copy(input, input + 9, mat.data());
    
    Matrix<double> rmat(3, 3);
    gen_qr(mat, rmat, input);
    
    double correct_q[3][3] = {
        {0.        , -0.4472136 , -0.89442719},
        {0.91287093,  0.36514837, -0.18257419},
        {0.40824829, -0.81649658,  0.40824829}
    };
    
    for (uint8_t i = 0; i < 3; i++) {
        for (uint8_t j = 0; j < 3; j++) {
            REQUIRE(mat(i, j) == Approx(correct_q[i][j]).margin(1e-7));
        }
    }
    
    double correct_r[3][3] = {
        {-6.70820393, 0, 0},
        {-8.04984472, 1.09544512, 0},
        {-10.2859127 ,   2.00831604,   0.40824829}
    };
    
    for (uint8_t i = 0; i < 3; i++) {
        for (uint8_t j = 0; j <= i; j++) {
            REQUIRE(rmat(i, j) == Approx(correct_r[i][j]).margin(1e-7));
        }
    }
}
