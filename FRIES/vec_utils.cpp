//
//  vec_utils.cpp
/*! \file
 *
 * \brief Definition of a function for compressing multiple vectors within a DistVec object
 */

#include "vec_utils.hpp"

void compress_vecs(DistVec<double> &vectors, size_t start_idx, size_t end_idx, unsigned int compress_size,
                  std::vector<size_t> &srt_scratch, std::vector<bool> &keep_scratch,
                  std::vector<bool> &del_arr, std::mt19937 &rn_gen) {
    for (uint16_t vec_idx = start_idx; vec_idx < end_idx; vec_idx++) {
        vectors.set_curr_vec_idx(vec_idx);
        piv_comp_parallel(vectors.values(), vectors.curr_size(), compress_size, srt_scratch, keep_scratch, rn_gen);

        for (size_t idx = 0; idx < vectors.curr_size(); idx++) {
            if (keep_scratch[idx]) {
                keep_scratch[idx] = 0;
            }
            else {
                del_arr[idx] = false;
            }
        }
    }
    for (size_t det_idx = 0; det_idx < vectors.curr_size(); det_idx++) {
        if (del_arr[det_idx]) {
            vectors.del_at_pos(det_idx);
        }
        del_arr[det_idx] = true;
    }
}


void compress_vecs_sys(DistVec<double> &vectors, size_t start_idx, size_t end_idx, unsigned int compress_size,
                  std::vector<size_t> &srt_scratch, std::vector<bool> &keep_scratch,
                  std::vector<bool> &del_arr, std::mt19937 &rn_gen) {
    int n_procs = 1;
    int proc_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    double norms[n_procs];
    double glob_norm;
    for (uint16_t vec_idx = start_idx; vec_idx < end_idx; vec_idx++) {
        vectors.set_curr_vec_idx(vec_idx);
        uint32_t n_samp = compress_size;
        norms[proc_rank] = find_preserve(vectors.values(), srt_scratch, keep_scratch, vectors.curr_size(), &n_samp, &glob_norm);
        double rn_sys = 0;
        if (proc_rank == 0) {
            rn_sys = rn_gen() / (1. + UINT32_MAX);
        }
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, norms, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        sys_comp(vectors.values(), vectors.curr_size(), norms, n_samp, keep_scratch, rn_sys);

        for (size_t idx = 0; idx < vectors.curr_size(); idx++) {
            if (keep_scratch[idx]) {
                keep_scratch[idx] = 0;
            }
            else {
                del_arr[idx] = false;
            }
        }
    }
    for (size_t det_idx = 0; det_idx < vectors.curr_size(); det_idx++) {
        if (del_arr[det_idx]) {
            vectors.del_at_pos(det_idx);
        }
        del_arr[det_idx] = true;
    }
}


void compress_vecs_multi(DistVec<double> &vectors, size_t start_idx, size_t end_idx, unsigned int compress_size,
                  std::vector<size_t> &srt_scratch, std::vector<bool> &keep_scratch,
                  std::vector<bool> &del_arr, std::mt19937 &rn_gen) {
    int n_procs = 1;
    int proc_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    double norms[n_procs];
    double loc_norm;
    double glob_norm;
    std::vector<uint16_t> counts(vectors.curr_size());
    
    uint16_t loc_samples[n_procs];
    uint16_t loc_nsamp;
    
    for (uint16_t vec_idx = start_idx; vec_idx < end_idx; vec_idx++) {
        vectors.set_curr_vec_idx(vec_idx);
        loc_norm = vectors.local_norm();
        glob_norm = sum_mpi(loc_norm, proc_rank, n_procs);
        MPI_Gather(&loc_norm, 1, MPI_DOUBLE, norms, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        for (size_t idx = 0; idx < vectors.curr_size(); idx++) {
            *vectors[idx] /= glob_norm;
            keep_scratch[idx] = *vectors[idx] > 0;
            *vectors[idx] = fabs(*vectors[idx]);
        }
        if (proc_rank == 0) {
            for (int proc_idx = 0; proc_idx < n_procs; proc_idx++) {
                norms[proc_idx] /= glob_norm;
            }
            unsigned int norm_aliases[n_procs];
            setup_alias(norms, norm_aliases, norms, n_procs);
            std::fill(loc_samples, loc_samples + n_procs, 0);
            sample_alias(norm_aliases, norms, n_procs, loc_samples, compress_size, rn_gen);
        }
        MPI_Scatter(loc_samples, 1, MPI_UINT16_T, &loc_nsamp, 1, MPI_UINT16_T, 0, MPI_COMM_WORLD);
        
        unsigned int *aliases = (unsigned int *)srt_scratch.data();
        setup_alias(vectors.values(), aliases, vectors.values(), vectors.curr_size());
        std::fill(counts.begin(), counts.end(), 0);
        sample_alias(aliases, vectors.values(), vectors.curr_size(), counts.data(), (unsigned int) loc_nsamp, rn_gen);
        
        for (size_t idx = 0; idx < vectors.curr_size(); idx++) {
            *vectors[idx] = glob_norm * counts[idx] * (keep_scratch[idx] ? 1 : -1) / compress_size;
            if (counts[idx] != 0) {
                del_arr[idx] = false;
            }
        }
    }
    for (size_t det_idx = 0; det_idx < vectors.curr_size(); det_idx++) {
        if (del_arr[det_idx]) {
            vectors.del_at_pos(det_idx);
        }
        del_arr[det_idx] = true;
    }
}
