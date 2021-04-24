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
