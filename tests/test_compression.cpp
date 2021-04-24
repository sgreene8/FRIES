/*! \file
 *
 * \brief Tests for stochastic compression functions
 */

#include "catch.hpp"
#include "inputs.hpp"
#include <fstream>
#include <FRIES/compress_utils.hpp>
#include <iomanip>

TEST_CASE("Test alias method", "[alias]") {
    using namespace test_inputs;
    double probs[] = {0.10125, 0.05625, 0.0875 , 0.03   , 0.095  , 0.05375, 0.095  ,
        0.0875 , 0.0625 , 0.33125};
    unsigned int n_states = 10;
    unsigned int n_samp = 30;
    unsigned int n_iter = 10000;

    double alias_probs[n_states];
    unsigned int aliases[n_states];
    setup_alias(probs, aliases, alias_probs, n_states);

    std::mt19937 mt_obj(0);
    uint8_t samples[n_samp];

    unsigned int cumu_samp[n_states];
    unsigned int iter_idx, samp_idx;
    std::string buf = out_path.append("alias.txt");
    std::ofstream cumu_f(buf);
    cumu_f << std::setprecision(10);
    REQUIRE(cumu_f.is_open() == true);
    for (samp_idx = 0; samp_idx < n_states; samp_idx++) {
        cumu_samp[samp_idx] = 0;
    }

    for (iter_idx = 0; iter_idx < n_iter; iter_idx++) {
        sample_alias(aliases, alias_probs, n_states, samples, n_samp, 1, mt_obj);
        for (samp_idx = 0; samp_idx < n_samp; samp_idx++) {
            cumu_samp[samples[samp_idx]]++;
        }
        for (samp_idx = 0; samp_idx < n_states; samp_idx++) {
            cumu_f << cumu_samp[samp_idx] / (iter_idx + 1.) / n_samp - probs[samp_idx] << ',';
        }
        cumu_f << '\n';
    }

    double max_diff = 0;
    double difference;
    for (samp_idx = 0; samp_idx < n_states; samp_idx++) {
        difference = fabs(cumu_samp[samp_idx] / 1. / n_samp / n_iter - probs[samp_idx]);
        if (difference > max_diff) {
            max_diff = difference;
        }
    }
    REQUIRE(max_diff == Approx(0).margin(1e-3));

    cumu_f.close();
}


TEST_CASE("Test that compression does nothing when target number of nonzeros is greater than input number of nonzeros", "[comp_preserve]") {
    int n_procs = 1;
    int proc_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    
    unsigned int input_len = 10;
    std::vector<double> input1(input_len);
    std::vector<double> input2(input_len);
    std::vector<bool> vec_keep(input_len, false);
    std::vector<size_t> vec_srt(input_len);
    
    unsigned int tot_samp = input_len * n_procs + 1;
    
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 mt_obj((unsigned int)seed);
    
    for (size_t idx = 0; idx < input_len; idx++) {
        input1[idx] = mt_obj() / (1. + UINT32_MAX);
        vec_srt[idx] = idx;
    }
    std::copy(input1.begin(), input1.end(), input2.begin());
    
    double norm_before;
    double norms[n_procs];
    norms[proc_rank] = find_preserve(input1.data(), vec_srt, vec_keep, input_len, &tot_samp, &norm_before);
    
    for (size_t el_idx = 0; el_idx < input_len; el_idx++) {
        REQUIRE(vec_keep[el_idx]);
    }
    REQUIRE(norms[proc_rank] == Approx(0).margin(1e-5));
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, norms, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    
    sys_comp(input1.data(), input_len, norms, tot_samp, vec_keep, mt_obj() / (1. + UINT32_MAX));
    
    for (size_t el_idx = 0; el_idx < input_len; el_idx++) {
        REQUIRE(input1[el_idx] == Approx(input2[el_idx]).margin(1e-5));
        REQUIRE(!vec_keep[el_idx]);
    }
    
    std::fill(vec_keep.begin(), vec_keep.end(), true);
    sys_comp_serial(input1.data(), input_len, 0, 1, 0, vec_keep, 0);
    
    for (size_t el_idx = 0; el_idx < input_len; el_idx++) {
        REQUIRE(input1[el_idx] == Approx(input2[el_idx]).margin(1e-5));
        REQUIRE(!vec_keep[el_idx]);
    }
    
    std::fill(vec_keep.begin(), vec_keep.end(), true);
    piv_samp_serial(input1.data(), input_len, 0, 0, vec_keep, mt_obj);
    
    for (size_t el_idx = 0; el_idx < input_len; el_idx++) {
        REQUIRE(input1[el_idx] == Approx(input2[el_idx]).margin(1e-5));
        REQUIRE(!vec_keep[el_idx]);
    }
}

