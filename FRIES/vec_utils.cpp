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
    int n_procs = 1;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double loc_norms[n_procs];
    for (uint16_t vec_idx = start_idx; vec_idx < end_idx; vec_idx++) {
        double glob_norm;
        unsigned int n_samp = compress_size;
        vectors.set_curr_vec_idx(vec_idx);
        loc_norms[rank] = find_preserve(vectors.values(), srt_scratch, keep_scratch, vectors.curr_size(), &n_samp, &glob_norm);
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, loc_norms, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        glob_norm = 0;
        for (uint16_t proc_idx = 0; proc_idx < n_procs; proc_idx++) {
            glob_norm += loc_norms[proc_idx];
        }

        uint32_t loc_samp = piv_budget(loc_norms, n_samp, rn_gen);
        uint32_t check_tot = sum_mpi((int)loc_samp, rank, n_procs);
        if (check_tot != n_samp) {
            std::stringstream msg;
            msg << "After pivotal budgeting, total number of elements across all processes (" << check_tot << ") does not equal input number of samples (" << n_samp << ")";
            throw std::runtime_error(msg.str());
        }
        double new_norm = adjust_probs(vectors.values(), vectors.curr_size(), &loc_samp, n_samp * loc_norms[rank] / glob_norm, n_samp, glob_norm, keep_scratch);
        piv_comp_serial(vectors.values(), vectors.curr_size(), new_norm, loc_samp, keep_scratch, rn_gen);

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
