/*! \file
 *
 * \brief Tests for stochastic compression functions
 */

#include "catch.hpp"
#include "inputs.hpp"
#include <fstream>
#include <FRIES/compress_utils.hpp>

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


TEST_CASE("Test calculation of observables from systematic sampling", "[sys_obs]") {
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 mt_obj((unsigned int)seed);
    
    unsigned int input_len = 10;
    size_t num_rns = 10;
    double input_vec[input_len];
    double tmp_vec[input_len];
    std::vector<size_t> vec_srt(input_len);
    for (size_t el_idx = 0; el_idx < input_len; el_idx++) {
        vec_srt[el_idx] = el_idx;
    }
    std::vector<bool> vec_keep1(input_len, false);
    std::vector<bool> vec_keep2(input_len, false);
    std::function<void(size_t, double *)> obs_fxn = [](size_t idx, double *res) {
        *res = idx + 1;
    };
    Matrix<double> obs_vals(num_rns, 1);
    
    int n_procs = 1;
    int proc_rank = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    double norms[n_procs];
    double norms2[n_procs];
    
    for (size_t test_idx = 0; test_idx < 10; test_idx++) {
        unsigned int n_samp = (test_idx % (input_len / 2)) + 1;
        double tot_norm;
        for (size_t el_idx = 0; el_idx < input_len; el_idx++) {
            input_vec[el_idx] = mt_obj() / (1. + UINT32_MAX);
            vec_keep1[el_idx] = false;
        }
        norms[proc_rank] = find_preserve(input_vec, vec_srt, vec_keep1, input_len, &n_samp, &tot_norm);

        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, norms, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        sys_obs(input_vec, input_len, norms, n_samp, vec_keep1, obs_fxn, obs_vals);
        
        for (size_t rn_idx = 0; rn_idx < num_rns; rn_idx++) {
            for (size_t el_idx = 0; el_idx < input_len; el_idx++) {
                tmp_vec[el_idx] = input_vec[el_idx];
                vec_keep2[el_idx] = vec_keep1[el_idx];
            }
            std::copy(norms, norms + n_procs, norms2);
            double rn = rn_idx / (double)num_rns;
            sys_comp(tmp_vec, input_len, norms2, n_samp, vec_keep2, rn);
            
            double comp_obs = 0;
            for (size_t el_idx = 0; el_idx < input_len; el_idx++) {
                double tmp_obs;
                obs_fxn(el_idx, &tmp_obs);
                comp_obs += tmp_obs * tmp_vec[el_idx] * tmp_vec[el_idx];
            }
            if (fabs(comp_obs - obs_vals(rn_idx, 0)) > 1e-7) {
                printf("Observable-based systematic compression failed for rn = %lf, n_samp = %lu\nVector:\n", rn, (test_idx % (input_len / 2)) + 1);
                for (size_t el_idx = 0; el_idx < input_len; el_idx++) {
                    printf("%lf->%lf\n", input_vec[el_idx], tmp_vec[el_idx]);
                }
                REQUIRE(comp_obs == Approx(obs_vals(rn_idx, 0)).margin(1e-7));
            }
        }
    }
}


TEST_CASE("Test systematic sampling with arbitrary distribution", "[sys_arbitrary]") {
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 mt_obj((unsigned int)seed);
    
    unsigned int input_len = 10;
    double input_vec[input_len];
    double tmp_vec[input_len];
    std::vector<size_t> vec_srt(input_len);
    for (size_t el_idx = 0; el_idx < input_len; el_idx++) {
        vec_srt[el_idx] = el_idx;
    }
    std::vector<bool> vec_keep1(input_len, false);
    std::vector<bool> vec_keep2(input_len, false);
    size_t num_rns = 10;
    double uni_probs[num_rns];
    for (size_t prob_idx = 0; prob_idx < num_rns; prob_idx++) {
        uni_probs[prob_idx] = 1. / num_rns;
    }
    
    unsigned int n_samp = input_len / 2;
    double tot_norm;
    double samp_norm = find_preserve(input_vec, vec_srt, vec_keep1, input_len, &n_samp, &tot_norm);
    for (size_t el_idx = 0; el_idx < input_len; el_idx++) {
        tmp_vec[el_idx] = input_vec[el_idx];
        vec_keep2[el_idx] = vec_keep1[el_idx];
    }
    double tmp_norm = samp_norm;
    double rn = 0.5107590879779309;
    sys_comp_nonuni(input_vec, input_len, &samp_norm, n_samp, vec_keep1, uni_probs, num_rns, rn);
    
    sys_comp(tmp_vec, input_len, &tmp_norm, n_samp, vec_keep2, rn);
    
    for (size_t el_idx = 0; el_idx < input_len; el_idx++) {
        REQUIRE(input_vec[el_idx] == Approx(tmp_vec[el_idx]).margin(1e-7));
    }
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
    piv_comp_serial(input1.data(), input_len, 0, 0, vec_keep, mt_obj);
    
    for (size_t el_idx = 0; el_idx < input_len; el_idx++) {
        REQUIRE(input1[el_idx] == Approx(input2[el_idx]).margin(1e-5));
        REQUIRE(!vec_keep[el_idx]);
    }
}

