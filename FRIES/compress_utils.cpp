/*! \file
 *
 * \brief Utilities for compressing vectors stochastically using the FRI
 * framework
 */

#include "compress_utils.hpp"
#include <cstdio>
#include <cstdint>
#include <climits>
#include <algorithm>
#include <FRIES/det_store.h>
#include <functional>
#include <iostream>
#include <sstream>


int round_binomially(double p, unsigned int n, std::mt19937 &mt_obj) {
    int flr = floor(p);
    double prob = p - flr;
    int ret_val = flr * n;
    for (unsigned int i = 0; i < n; i++) {
        ret_val += (mt_obj() / (1. + UINT32_MAX)) < prob;
    }
    return ret_val;
}

double find_preserve(double *values, std::vector<size_t> &srt_idx, std::vector<bool> &keep_idx,
                     size_t count, unsigned int *n_samp, double *global_norm) {
    return find_preserve(values, srt_idx, keep_idx, count, n_samp, global_norm, MPI_COMM_WORLD);
}

double find_preserve(double *values, std::vector<size_t> &srt_idx, std::vector<bool> &keep_idx,
                     size_t count, unsigned int *n_samp, double *global_norm, MPI_Comm comm)  {
    double loc_one_norm = 0;
    double glob_one_norm = 0;
    size_t heap_count = count;
    for (size_t det_idx = 0; det_idx < count; det_idx++) {
        loc_one_norm += fabs(values[det_idx]);
    }
    int proc_rank = 0;
    int n_procs = 1;
    MPI_Comm_rank(comm, &proc_rank);
    MPI_Comm_size(comm, &n_procs);
    
    auto val_compare = [values](size_t i, size_t j){return fabs(values[i]) < fabs(values[j]); };
    std::make_heap(srt_idx.begin(), srt_idx.begin() + heap_count, val_compare);
    int loc_sampled, glob_sampled = 1;
    int keep_going = 1;
    
    double el_magn = 0;
    size_t max_idx;
    *global_norm = sum_mpi(loc_one_norm, proc_rank, n_procs, comm);
    bool recalc_norm = false;
    while (glob_sampled > 0) {
        glob_one_norm = sum_mpi(loc_one_norm, proc_rank, n_procs, comm);
        loc_sampled = 0;
        while (keep_going && heap_count > 0) {
            max_idx = srt_idx[0];
            el_magn = fabs(values[max_idx]);
            if (el_magn >= glob_one_norm / (*n_samp - loc_sampled)) {
                keep_idx[max_idx] = 1;
                loc_sampled++;
                loc_one_norm -= el_magn;
                glob_one_norm -= el_magn;
                
                heap_count--;
                if (heap_count) {
                    std::pop_heap(srt_idx.begin(), srt_idx.begin() + heap_count + 1, val_compare);
                }
                else {
                    keep_going = 0;
                }
            }
            else{
                keep_going = 0;
            }
        }
        glob_sampled = sum_mpi(loc_sampled, proc_rank, n_procs, comm);
        (*n_samp) -= glob_sampled;
        if (glob_sampled == 0 && !recalc_norm) {
            loc_one_norm = 0;
            for (size_t el_idx = 0; el_idx < count; el_idx++) {
                if (!keep_idx[el_idx]) {
                    loc_one_norm += fabs(values[el_idx]);
                }
            }
            glob_sampled = 1;
            recalc_norm = true;
        }
        else {
            recalc_norm = false;
        }
        keep_going = 1;
    }
    loc_one_norm = 0;
    if (glob_one_norm < 1e-9) {
        *n_samp = 0;
    }
    else {
        for (size_t det_idx = 0; det_idx < count; det_idx++) {
            if (!keep_idx[det_idx]) {
                loc_one_norm += fabs(values[det_idx]);
            }
        }
    }
    return loc_one_norm;
}

double seed_sys(double *norms, double *rn, unsigned int n_samp) {
    double lbound = 0;
    int n_procs = 1;
    int my_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    double global_norm;
    for (int proc_idx = 0; proc_idx < my_rank; proc_idx++) {
        lbound += norms[proc_idx];
    }
    global_norm = lbound;
    for (int proc_idx = my_rank; proc_idx < n_procs; proc_idx++) {
        global_norm += norms[proc_idx];
    }
    *rn *= global_norm / n_samp;
    *rn += global_norm / n_samp * (int)(lbound * n_samp / global_norm);
    if (*rn < lbound) {
        *rn += global_norm / n_samp;
    }
    return lbound;
}


double find_keep_sub(double *values, unsigned int *n_div,
                     const Matrix<double> &sub_weights, Matrix<bool> &keep_idx,
                     uint16_t *sub_sizes,
                     size_t count, unsigned int *n_samp, double *wt_remain) {
    double loc_one_norm = 0;
    double glob_one_norm = 0;
    for (size_t det_idx = 0; det_idx < count; det_idx++) {
        loc_one_norm += values[det_idx];
        wt_remain[det_idx] = values[det_idx];
    }
    int proc_rank = 0;
    int n_procs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    
    int loc_sampled, glob_sampled = 1;
    double sub_magn, sub_remain;
    int last_pass = 0;
    size_t n_sub = sub_weights.cols();
    size_t coarse_size = 8;
    uint8_t coarse_bool;
    size_t n_coarse = count / coarse_size;
    double coarse_weights[coarse_size];
    while (glob_sampled > 0) {
        glob_one_norm = sum_mpi(loc_one_norm, proc_rank, n_procs);
        if (glob_one_norm < 0) {
            break;
        }
        loc_sampled = 0;
        for (size_t coarse_idx = 0; coarse_idx <= n_coarse; coarse_idx++) {
            coarse_bool = 0;
            size_t fine_limit;
            if (coarse_idx == n_coarse) {
                fine_limit = count % coarse_size;
            }
            else {
                fine_limit = coarse_size;
            }
            double wt_factor = *n_samp - loc_sampled;
            double *wt_remain2 = &wt_remain[coarse_idx * coarse_size];
            double *values2 = &values[coarse_idx * coarse_size];
            unsigned int *ndiv2 = &n_div[coarse_idx * coarse_size];
            for (size_t fine_idx = 0; fine_idx < fine_limit; fine_idx++) {
                if (wt_remain2[fine_idx] > 0) {
                    coarse_weights[fine_idx] = values2[fine_idx] * wt_factor;
                    if (ndiv2[fine_idx] > 0) {
                        coarse_weights[fine_idx] /= ndiv2[fine_idx];
                    }
                    coarse_bool += (coarse_weights[fine_idx] >= glob_one_norm) << fine_idx;
                }
            }
            uint8_t n_true = byte_nums[coarse_bool];
            for (size_t fine_idx = 0; fine_idx < n_true; fine_idx++) {
                size_t det_idx = coarse_idx * coarse_size + byte_pos[coarse_bool][fine_idx];
                double el_magn = values[det_idx];
                if (n_div[det_idx] > 0) {
                    keep_idx(det_idx, 0) = 1;
                    wt_remain[det_idx] = 0;
                    loc_sampled += n_div[det_idx];
                    loc_one_norm -= el_magn;
                    glob_one_norm -= el_magn;
                    if (glob_one_norm < 0) {
                        break;
                    }
                }
                else {
                    sub_remain = 0;
                    const double *subwt_row = sub_weights[det_idx];
                    if (sub_sizes) {
                        n_sub = sub_sizes[det_idx];
                    }
                    double coarse_wt = coarse_weights[byte_pos[coarse_bool][fine_idx]];
                    
                    uint8_t *keep_row = keep_idx.row_ptr(det_idx);
                    for (size_t sub_coarse = 0; sub_coarse < (n_sub / 8); sub_coarse++) {
                        uint8_t not_kept = ~keep_row[sub_coarse];
                        uint8_t row_bool = 0;
                        uint8_t n_not_kept = byte_nums[not_kept];
                        for (size_t sub_fine = 0; sub_fine < n_not_kept; sub_fine++) {
                            uint8_t pos = byte_pos[not_kept][sub_fine];
                            sub_magn = coarse_wt * subwt_row[sub_coarse * 8 + pos];
                            if (sub_magn >= glob_one_norm && fabs(sub_magn) > 1e-12) {
                                row_bool += (1 << pos);
                                loc_sampled++;
                            }
                            else {
                                sub_remain += sub_magn;
                            }
                        }
                        keep_row[sub_coarse] |= row_bool;
                    }
                    if (n_sub % 8) {
                        size_t sub_coarse = n_sub / 8;
                        uint8_t not_kept = ~keep_row[sub_coarse];
                        not_kept &= (1 << (n_sub % 8)) - 1;
                        uint8_t row_bool = 0;
                        uint8_t n_not_kept = byte_nums[not_kept];
                        for (size_t sub_fine = 0; sub_fine < n_not_kept; sub_fine++) {
                            uint8_t pos = byte_pos[not_kept][sub_fine];
                            sub_magn = coarse_wt * subwt_row[sub_coarse * 8 + pos];
                            if (sub_magn >= glob_one_norm && fabs(sub_magn) > 1e-10) {
                                row_bool += (1 << pos);
                                loc_sampled++;
                            }
                            else {
                                sub_remain += sub_magn;
                            }
                        }
                        keep_row[sub_coarse] |= row_bool;
                    }
                    sub_remain /= wt_factor;
                    double change = wt_remain[det_idx] - sub_remain;
                    wt_remain[det_idx] = sub_remain;
                    loc_one_norm -= change;
                    glob_one_norm -= change;
                }
            }
        }
        glob_sampled = sum_mpi(loc_sampled, proc_rank, n_procs);
        (*n_samp) -= glob_sampled;
        
        if (last_pass && glob_sampled) {
            last_pass = 0;
        }
        if (glob_sampled == 0 && !last_pass) {
            last_pass = 1;
            glob_sampled = 1;
            loc_one_norm = 0;
            for (size_t det_idx = 0; det_idx < count; det_idx++) {
                loc_one_norm += wt_remain[det_idx];
            }
        }
    }
    loc_one_norm = 0;
    if (glob_one_norm / *n_samp < 1e-8) {
        *n_samp = 0;
    }
    else {
        for (size_t det_idx = 0; det_idx < count; det_idx++) {
            loc_one_norm += wt_remain[det_idx];
        }
    }
    return loc_one_norm;
}

void sys_comp(double *vec_vals, size_t vec_len, double *loc_norms,
              unsigned int n_samp, std::vector<bool> &keep_exact, double rand_num) {
    sys_comp(vec_vals, vec_len, loc_norms, n_samp, keep_exact, rand_num, MPI_COMM_WORLD);
}

void sys_comp(double *vec_vals, size_t vec_len, double *loc_norms,
              unsigned int n_samp, std::vector<bool> &keep_exact, double rand_num,
              MPI_Comm comm) {
    int n_procs = 1;
    int proc_rank = 0;
    double rn_sys = rand_num;
    MPI_Comm_size(comm, &n_procs);
    MPI_Comm_rank(comm, &proc_rank);
    MPI_Bcast(&rn_sys, 1, MPI_DOUBLE, 0, comm);
    double tmp_glob_norm = 0;
    for (int proc_idx = 0; proc_idx < n_procs; proc_idx++) {
        tmp_glob_norm += loc_norms[proc_idx];
    }
    
    double lbound;
    if (n_samp > 0) {
        lbound = seed_sys(loc_norms, &rn_sys, n_samp);
    }
    else {
        lbound = 0;
        rn_sys = INFINITY;
    }
    
    loc_norms[proc_rank] = 0;
    for (size_t det_idx = 0; det_idx < vec_len; det_idx++) {
        double tmp_val = vec_vals[det_idx];
        if (keep_exact[det_idx]) {
            loc_norms[proc_rank] += fabs(tmp_val);
            keep_exact[det_idx] = 0;
        }
        else if (tmp_val != 0) {
            lbound += fabs(tmp_val);
            if (rn_sys < lbound) {
                vec_vals[det_idx] = tmp_glob_norm / n_samp * ((tmp_val > 0) - (tmp_val < 0));
                loc_norms[proc_rank] += tmp_glob_norm / n_samp;
                rn_sys += tmp_glob_norm / n_samp;
            }
            else {
                vec_vals[det_idx] = 0;
                keep_exact[det_idx] = 1;
            }
        }
    }
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, loc_norms, 1, MPI_DOUBLE, comm);
}

void sys_comp_serial(double *vec_vals, size_t vec_len, double seg_norm, double sampl_val,
                     uint32_t n_samp, std::vector<bool> &keep_exact, double rand_num) {
    double sampl_unit = seg_norm / n_samp;
    double rn_sys = rand_num * sampl_unit;
    double lbound = 0;
    for (size_t det_idx = 0; det_idx < vec_len; det_idx++) {
        double tmp_val = vec_vals[det_idx];
        if (keep_exact[det_idx]) {
            keep_exact[det_idx] = 0;
        }
        else if (tmp_val != 0) {
            lbound += fabs(tmp_val);
            if (rn_sys < lbound) {
                vec_vals[det_idx] = sampl_val * ((tmp_val > 0) - (tmp_val < 0));
                rn_sys += sampl_unit;
            }
            else {
                vec_vals[det_idx] = 0;
                keep_exact[det_idx] = true;
            }
        }
    }
}


void piv_comp_serial(double *vec_vals, size_t vec_len, double seg_norm,
                     uint32_t n_samp, std::vector<bool> &keep_exact, std::mt19937 &mt_obj) {
    if (n_samp == 0) {
        for (size_t idx = 0; idx < vec_len; idx++) {
            if (keep_exact[idx]) {
                keep_exact[idx] = false;
            }
            else {
                vec_vals[idx] = 0;
                keep_exact[idx] = true;
            }
        }
        return;
    }
    double sampl_unit = seg_norm / n_samp;
    size_t vec_idx = 0;
    std::vector<double> sampl_el(2 * vec_len / n_samp);
    size_t resid_idx = 0;
    double cum_prob;
    sampl_el[0] = 0; // residual element
    uint32_t samp_so_far = 0;
    while (vec_idx < vec_len && samp_so_far < n_samp) {
        // build probability vector
        size_t prob_idx;
        size_t vec_offset = 0;
        cum_prob = sampl_el[0];
        for (prob_idx = 1; cum_prob < sampl_unit && (vec_idx + vec_offset) < vec_len; vec_offset++) {
            size_t vec_max = sampl_el.capacity();
            if (!keep_exact[vec_idx + vec_offset]) {
                if (prob_idx == vec_max) {
                    sampl_el.resize(vec_max * 2);
                }
                sampl_el[prob_idx] = fabs(vec_vals[vec_idx + vec_offset]);
                cum_prob += sampl_el[prob_idx];
                prob_idx++;
            }
        }
        size_t vec_max_offset = vec_offset - 1;
        if (vec_offset + vec_idx == vec_len) {
            vec_max_offset++;
        }
        double bn = cum_prob - sampl_unit;
        if (vec_offset + vec_idx != vec_len) { // If not at the last sampling interval
            prob_idx--;
            cum_prob -= sampl_el[prob_idx];
        }
        double an = sampl_unit - cum_prob;
        
        // sample H_n from among the units before the cross-border unit
        double rn = mt_obj() / (1. + UINT32_MAX) * cum_prob;
        cum_prob = 0;
        size_t Hn = 0;
        while (cum_prob < rn && Hn < prob_idx) {
            cum_prob += sampl_el[Hn];
            Hn++;
        }
        if (rn > 0) {
            Hn--;
        }
        if (Hn != 0 && vec_idx != 0) { // residual unit will never be sampled
            vec_vals[resid_idx] = 0;
            keep_exact[resid_idx] = true;
        }
        
        // Decide which sample to keep and which to pass on
        double pass_Hn_prob = an / (sampl_unit - bn); // probability of passing the sample H_n into the next sampling unit
        if (vec_offset + vec_idx == vec_len) {
            pass_Hn_prob = 0;
        }
        rn = mt_obj() / (1. + UINT32_MAX); // [0, 1) random number
        if (rn < pass_Hn_prob) { // cross-border unit is sampled and H_n is passed on
            prob_idx = 1;
            for (vec_offset = 0; vec_offset < vec_max_offset; vec_offset++) {
                if (!keep_exact[vec_idx + vec_offset]) {
                    if (prob_idx == Hn) { // pass it on
                        resid_idx = vec_idx + vec_offset;
                    }
                    else {
                        vec_vals[vec_idx + vec_offset] = 0;
                        keep_exact[vec_idx + vec_offset] = true;
                    }
                    prob_idx++;
                }
                else {
                    keep_exact[vec_idx + vec_offset] = false;
                }
            }
            // sample cross-border unit
            double tmp_val = vec_vals[vec_idx + vec_max_offset];
            vec_vals[vec_idx + vec_max_offset] = sampl_unit * ((tmp_val > 0) - (tmp_val < 0));
        }
        else { // H_n is sampled and cross-border unit is passed on
               // deal with residual unit
            if (Hn == 0) { // residual unit was sampled
                double tmp_val = vec_vals[resid_idx];
                vec_vals[resid_idx] = sampl_unit * ((tmp_val > 0) - (tmp_val < 0));
            }
            prob_idx = 1;
            for (vec_offset = 0; vec_offset < vec_max_offset; vec_offset++) {
                if (!keep_exact[vec_idx + vec_offset]) {
                    if (prob_idx != Hn) { // this element will never be sampled
                        vec_vals[vec_idx + vec_offset] = 0;
                        keep_exact[vec_idx + vec_offset] = true;
                    }
                    else { // this element is sampled
                        double tmp_val = vec_vals[vec_idx + vec_offset];
                        vec_vals[vec_idx + vec_offset] = sampl_unit * ((tmp_val > 0) - (tmp_val < 0));
                    }
                    prob_idx++;
                }
                else {
                    keep_exact[vec_idx + vec_offset] = false;
                }
            }
            // pass on cross-border unit
            resid_idx = vec_idx + vec_max_offset;
        }
        vec_idx += vec_offset + 1;
        sampl_el[0] = bn; // residual element
        samp_so_far++;
    }
    for (; vec_idx < vec_len; vec_idx++) {
        if (!keep_exact[vec_idx]) {
            vec_vals[vec_idx] = 0;
            keep_exact[vec_idx] = true;
        }
        else {
            keep_exact[vec_idx] = false;
        }
    }
    // Zero the residual element
    if (resid_idx < vec_len) {
        vec_vals[resid_idx] = 0;
        keep_exact[resid_idx] = true;
    }
}


void sys_comp_nonuni(double *vec_vals, size_t vec_len, double *loc_norms,
                     unsigned int n_samp, std::vector<bool> &keep_exact,
                     double *probs, size_t n_probs, double rand_num) {
    int n_procs = 1;
    int proc_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    double rn_sys = 0;
    if (proc_rank == 0) {
        double cum_prob = probs[0];
        size_t prob_idx = 1;
        for (; prob_idx < n_probs && cum_prob < rand_num; prob_idx++) {
            cum_prob += probs[prob_idx];
        }
        prob_idx--;
        rn_sys = (prob_idx + 1 - (cum_prob - rand_num) / probs[prob_idx]) / n_probs;
    }
    MPI_Bcast(&rn_sys, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double tmp_glob_norm = 0;
    for (int proc_idx = 0; proc_idx < n_procs; proc_idx++) {
        tmp_glob_norm += loc_norms[proc_idx];
    }
    double mag_before = 0;
    if (n_samp > 0) {
        mag_before = seed_sys(loc_norms, &rn_sys, n_samp);
    }
    else {
        mag_before = 0;
        rn_sys = INFINITY;
    }
    
    loc_norms[proc_rank] = 0;
    for (size_t det_idx = 0; det_idx < vec_len; det_idx++) {
        double tmp_val = vec_vals[det_idx];
        if (keep_exact[det_idx]) {
            loc_norms[proc_rank] += fabs(tmp_val);
            keep_exact[det_idx] = 0;
        }
        else if (tmp_val != 0) {
            mag_before += fabs(tmp_val);
            if (rn_sys < mag_before) {
                double left_rn_idx = (mag_before - fabs(tmp_val)) / (tmp_glob_norm / n_samp / n_probs);
                double right_rn_idx = mag_before / (tmp_glob_norm / n_samp / n_probs);
                size_t rn_idx = left_rn_idx;
                double pdf_frac;
                double pdf_integral = 0;
                if (rn_idx == (size_t)right_rn_idx) {
                    pdf_frac = right_rn_idx - left_rn_idx;
                }
                else {
                    pdf_frac = (size_t)(left_rn_idx + 1) - left_rn_idx;
                    pdf_integral = pdf_frac * probs[rn_idx % n_probs];
                    for (rn_idx++; rn_idx <= right_rn_idx - 1; rn_idx++) {
                        pdf_integral += probs[rn_idx % n_probs];
                    }
                    pdf_frac = right_rn_idx - (size_t)right_rn_idx;
                }
                pdf_integral += probs[rn_idx % n_probs] * pdf_frac;
                
                vec_vals[det_idx] = fabs(tmp_val) / pdf_integral * ((tmp_val > 0) - (tmp_val < 0));
                loc_norms[proc_rank] += fabs(tmp_val) / pdf_integral;
                rn_sys += tmp_glob_norm / n_samp;
            }
            else {
                vec_vals[det_idx] = 0;
                keep_exact[det_idx] = 1;
            }
        }
    }
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, loc_norms, 1, MPI_DOUBLE, MPI_COMM_WORLD);
}


void sys_obs(double *vec_vals, size_t vec_len, double *loc_norms, unsigned int n_samp,
             std::vector<bool> &keep_exact, std::function<void(size_t, double *)> obs,
             Matrix<double> &obs_vals) {
    int n_procs = 1;
    int proc_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    double glob_norm = 0;
    for (int proc_idx = 0; proc_idx < n_procs; proc_idx++) {
        glob_norm += loc_norms[proc_idx];
    }
    
    obs_vals.zero();
    size_t n_obs = obs_vals.cols();
    std::vector<double> obs_tmp(n_obs);
    
    double mag_before = 0;
    for (int proc_idx = 0; proc_idx < proc_rank; proc_idx++) {
        mag_before += loc_norms[proc_idx];
    }
    size_t num_rns = obs_vals.rows();
    
    double left_rn_idx = mag_before * n_samp * num_rns / glob_norm;
    
    for (size_t el_idx = 0; el_idx < vec_len; el_idx++) {
        std::fill(obs_tmp.begin(), obs_tmp.end(), 0);
        obs(el_idx, obs_tmp.data());
        double tmp_val = fabs(vec_vals[el_idx]);
        if (keep_exact[el_idx]) {
            for (size_t rn_idx = 0; rn_idx < num_rns; rn_idx++) {
                for (size_t obs_idx = 0; obs_idx < n_obs; obs_idx++) {
                    obs_vals(rn_idx, obs_idx) += obs_tmp[obs_idx] * tmp_val * tmp_val;
                }
            }
        }
        else {
            mag_before += tmp_val;
            double right_rn_idx = mag_before * n_samp * num_rns / glob_norm;
            size_t rn_idx = left_rn_idx;
            if (fabs(rn_idx - left_rn_idx) != 0) {
                rn_idx++;
            }
            if (fabs(right_rn_idx - n_samp * num_rns) < 1e-8) { // correct for floating-point error
                right_rn_idx = n_samp * num_rns - 0.5;
            }
            for (; rn_idx < right_rn_idx; rn_idx++) {
                for (size_t obs_idx = 0; obs_idx < n_obs; obs_idx++) {
                    obs_vals(rn_idx % num_rns, obs_idx) += obs_tmp[obs_idx] * glob_norm / n_samp * glob_norm / n_samp;
                }
            }
            left_rn_idx = right_rn_idx;
        }
    }
}


uint32_t sys_budget(double *loc_norms, uint32_t n_samp, double rand_num) {
    int n_procs = 1;
    int proc_rank = 0;
    double rn_sys = rand_num;
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Bcast(&rn_sys, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double tmp_glob_norm = 0;
    for (int proc_idx = 0; proc_idx < n_procs; proc_idx++) {
        tmp_glob_norm += loc_norms[proc_idx];
    }
    
    if (n_samp > 0) {
        double lbound = seed_sys(loc_norms, &rn_sys, n_samp);
        double num_int = (lbound + loc_norms[proc_rank] - rn_sys) * n_samp / tmp_glob_norm;
        int32_t ret_num = num_int + 1;
        if ((num_int - (int)num_int) < 1e-8) {
            ret_num--;
        }
        if (ret_num < 0) {
            ret_num = 0;
        }
        return ret_num;
    }
    else {
        return 0;
    }
}


uint32_t piv_budget(double *loc_norms, uint32_t n_samp, std::mt19937 &mt_obj) {
    return piv_budget(loc_norms, n_samp, mt_obj, MPI_COMM_WORLD);
}

uint32_t piv_budget(double *loc_norms, uint32_t n_samp, std::mt19937 &mt_obj,
                    MPI_Comm comm) {
    int n_procs = 1;
    int proc_rank = 0;
    MPI_Comm_size(comm, &n_procs);
    MPI_Comm_rank(comm, &proc_rank);
    
    std::vector<uint32_t> budgets(n_procs);
    uint32_t proc_budget;
    
    if (proc_rank == 0) {
        double glob_norm = 0;
        for (int proc_idx = 0; proc_idx < n_procs; proc_idx++) {
            glob_norm += loc_norms[proc_idx];
        }
        
        uint32_t tot_budget = 0;
        std::vector<double> weights(n_procs);
        uint32_t n_nonz = 0;
        for (int proc_idx = 0; proc_idx < n_procs; proc_idx++) {
            budgets[proc_idx] = loc_norms[proc_idx] / glob_norm * n_samp;
            tot_budget += budgets[proc_idx];
            weights[proc_idx] = loc_norms[proc_idx] - budgets[proc_idx] * glob_norm / n_samp;
            if (weights[proc_idx] < 1e-12) {
                weights[proc_idx] = 0;
            }
            if (weights[proc_idx] > 0) {
                n_nonz++;
            }
        }
        if (n_nonz == (n_samp - tot_budget)) {
            for (int proc_idx = 0; proc_idx < n_procs; proc_idx++) {
                if (weights[proc_idx] > 0) {
                    budgets[proc_idx]++;
                }
            }
            tot_budget = n_samp;
        }
        
        if (tot_budget < n_samp) {
            std::vector<bool> keep(n_procs, false);
            piv_comp_serial(weights.data(), n_procs, glob_norm * (n_samp - tot_budget) / n_samp, n_samp - tot_budget, keep, mt_obj);
            for (int proc_idx = 0; proc_idx < n_procs; proc_idx++) {
                if (weights[proc_idx] > 0) {
                    budgets[proc_idx]++;
                }
            }
        }
    }
    MPI_Scatter(budgets.data(), 1, MPI_UINT32_T, &proc_budget, 1, MPI_UINT32_T, 0, comm);
    return proc_budget;
}

double adjust_probs(double *vec_vals, size_t vec_len, uint32_t *n_samp_loc,
                    double exp_nsamp_loc, uint32_t n_samp_tot, double tot_norm,
                    std::vector<bool> &keep_exact) {
    bool el_too_big = false;
    double ceil = ceill(exp_nsamp_loc);
    double resid = exp_nsamp_loc - (unsigned int)exp_nsamp_loc;
    double sampling_unit = tot_norm / n_samp_tot;
    double loc_norm = exp_nsamp_loc * sampling_unit;
    for (size_t vec_idx = 0; vec_idx < vec_len; vec_idx++) {
        if (!keep_exact[vec_idx] && fabs(vec_vals[vec_idx]) >= loc_norm / ceil) {
            el_too_big = true;
            break;
        }
    }
    if (el_too_big) {
        double counter = exp_nsamp_loc;
        if (*n_samp_loc > exp_nsamp_loc) {
            for (size_t vec_idx = 0; vec_idx < vec_len; vec_idx++) {
                if (!keep_exact[vec_idx]) {
                    int8_t sign = 2 * (vec_vals[vec_idx] > 0) - 1;
                    double pi = fabs(vec_vals[vec_idx]) / sampling_unit;
                    if (pi < resid) {
                        counter += pi / resid - pi;
                        vec_vals[vec_idx] /= resid;
                    }
                    else {
                        counter -= pi;
                        vec_vals[vec_idx] = sign * sampling_unit;
                        keep_exact[vec_idx] = true;
                        (*n_samp_loc)--;
                    }
                    if (counter >= *n_samp_loc) {
                        vec_vals[vec_idx] += sign * sampling_unit * (*n_samp_loc - counter);
                        break;
                    }
                }
            }
        }
        else {
            for (size_t vec_idx = 0; vec_idx < vec_len; vec_idx++) {
                if (!keep_exact[vec_idx]) {
                    int8_t sign = 2 * (vec_vals[vec_idx] > 0) - 1;
                    double pi = fabs(vec_vals[vec_idx]) / sampling_unit;
                    if (pi > resid) {
                        double quotient = (pi - resid) / (1 - resid);
                        counter += quotient - pi;
                        vec_vals[vec_idx] = sign * quotient * sampling_unit;
                    }
                    else {
                        counter -= pi;
                        vec_vals[vec_idx] = 0;
                    }
                    if (counter <= *n_samp_loc) {
                        vec_vals[vec_idx] += sign * sampling_unit * (*n_samp_loc - counter);
                        break;
                    }
                }
            }
        }
        return *n_samp_loc * loc_norm / exp_nsamp_loc;
    }
    else {
        return loc_norm;
    }
}


void adjust_shift(double *shift, double one_norm, double *last_norm,
                  double target_norm, double damp_factor) {
    if (*last_norm) {
        *shift -= damp_factor * log(one_norm / *last_norm);
        *last_norm = one_norm;
    }
    if (*last_norm == 0 && one_norm > target_norm) {
        *last_norm = one_norm;
    }
}

size_t sys_sub(double *values, unsigned int *n_div,
               const Matrix<double> &sub_weights, Matrix<bool> &keep_idx,
               uint16_t *sub_sizes,
               size_t count, unsigned int n_samp, double *wt_remain,
               double *loc_norms, double rand_num, double *new_vals,
               size_t new_idx[][2]) {
    int n_procs = 1;
    int proc_rank = 0;
    double rn_sys = rand_num;
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Bcast(&rn_sys, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double tmp_glob_norm = 0;
    for (int proc_idx = 0; proc_idx < n_procs; proc_idx++) {
        tmp_glob_norm += loc_norms[proc_idx];
    }
    
    double lbound;
    if (n_samp > 0) {
        lbound = seed_sys(loc_norms, &rn_sys, n_samp);
    }
    else {
        lbound = 0;
        rn_sys = INFINITY;
    }
    
    loc_norms[proc_rank] = 0;
    size_t sub_idx;
    double tmp_val;
    size_t num_new = 0;
    double sub_lbound;
    size_t n_sub = keep_idx.cols();
    for (size_t wt_idx = 0; wt_idx < count; wt_idx++) {
        tmp_val = values[wt_idx];
        if (tmp_val == 0) {
            continue;
        }
        lbound += wt_remain[wt_idx];
        if (n_div[wt_idx] > 0) {
            if (keep_idx(wt_idx, 0)) {
                keep_idx(wt_idx, 0) = 0;
                for (sub_idx = 0; sub_idx < n_div[wt_idx]; sub_idx++) {
                    new_vals[num_new] = tmp_val / n_div[wt_idx];
                    new_idx[num_new][0] = wt_idx;
                    new_idx[num_new][1] = sub_idx;
                    num_new++;
                }
                loc_norms[proc_rank] += tmp_val;
            }
            else if (tmp_val != 0) {
                while (rn_sys < lbound) {
                    sub_idx = (lbound - rn_sys) * n_div[wt_idx] / tmp_val;
                    if (sub_idx < n_div[wt_idx]) {
                        new_vals[num_new] = tmp_glob_norm / n_samp;
                        new_idx[num_new][0] = wt_idx;
                        new_idx[num_new][1] = sub_idx;
                        num_new++;
                        loc_norms[proc_rank] += tmp_glob_norm / n_samp;
                    }
                    rn_sys += tmp_glob_norm / n_samp;
                }
            }
        }
        else if (wt_remain[wt_idx] < tmp_val || rn_sys < lbound) {
            loc_norms[proc_rank] += (tmp_val - wt_remain[wt_idx]); // add kept weight
            sub_lbound = lbound - wt_remain[wt_idx];
            if (sub_sizes) {
                n_sub = sub_sizes[wt_idx];
            }
            for (sub_idx = 0; sub_idx < n_sub; sub_idx++) {
                if (keep_idx(wt_idx, sub_idx) && sub_weights[wt_idx][sub_idx] != 0) {
                    new_vals[num_new] = tmp_val * sub_weights[wt_idx][sub_idx];
                    new_idx[num_new][0] = wt_idx;
                    new_idx[num_new][1] = sub_idx;
                    num_new++;
                }
                else {
                    sub_lbound += tmp_val * sub_weights[wt_idx][sub_idx];
                    if (rn_sys < sub_lbound && sub_weights[wt_idx][sub_idx] != 0) {
                        new_vals[num_new] = tmp_glob_norm / n_samp;
                        new_idx[num_new][0] = wt_idx;
                        new_idx[num_new][1] = sub_idx;
                        num_new++;
                        loc_norms[proc_rank] += tmp_glob_norm / n_samp;
                        rn_sys += tmp_glob_norm / n_samp;
                    }
                }
                keep_idx(wt_idx, sub_idx) = 0;
            }
        }
    }
    return num_new;
}


size_t comp_sub(double *values, size_t count, unsigned int *n_div,
                Matrix<double> &sub_weights, Matrix<bool> &keep_idx,
                uint16_t *sub_sizes,
                unsigned int n_samp, double *wt_remain, double rand_num,
                double *new_vals, size_t new_idx[][2]) {
    int proc_rank = 0;
    int n_procs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Bcast(&rand_num, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    unsigned int tmp_nsamp = n_samp;
    double loc_norms[n_procs];
    if (keep_idx.cols() != sub_weights.cols()) {
        std::stringstream msg;
        msg << "Error in comp_sub: column dimension of sub_weights (" << sub_weights.cols() << ") does not equal column dimension of keep_idx (" << keep_idx.cols() << "\n";
        throw std::runtime_error(msg.str());
        return 0;
    }
    
    loc_norms[proc_rank] = find_keep_sub(values, n_div, sub_weights, keep_idx, sub_sizes, count, &tmp_nsamp, wt_remain);
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, loc_norms, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    return sys_sub(values, n_div, sub_weights, keep_idx, sub_sizes, count, tmp_nsamp, wt_remain, loc_norms, rand_num, new_vals, new_idx);
}


void setup_alias(double *probs, unsigned int *aliases, double *alias_probs,
                 size_t n_states) {
    size_t n_s = 0;
    size_t n_b = 0;
    unsigned int smaller[n_states];
    unsigned int bigger[n_states];
    unsigned int s, b;
    for (unsigned int i = 0; i < n_states; i++) {
        aliases[i] = i;
        alias_probs[i] = n_states * probs[i];
        if (alias_probs[i] < 1) {
            smaller[n_s] = i;
            n_s++;
        }
        else {
            bigger[n_b] = i;
            n_b++;
        }
    }
    while (n_s > 0 && n_b > 0) {
        s = smaller[n_s - 1];
        b = bigger[n_b - 1];
        aliases[s] = b;
        alias_probs[b] += alias_probs[s] - 1;
        if (alias_probs[b] < 1) {
            smaller[n_s - 1] = b;
            n_b--;
        }
        else {
            n_s--;
        }
    }
}


void sample_alias(unsigned int *aliases, double *alias_probs, size_t n_states,
                  uint8_t *samples, unsigned int n_samp, size_t samp_int,
                  std::mt19937 &mt_obj) {
    uint8_t chosen_idx;
    for (unsigned int samp_idx = 0; samp_idx < n_samp; samp_idx++) {
        chosen_idx = mt_obj() / (1. + UINT32_MAX) * n_states;
        if (mt_obj() / (1. + UINT32_MAX) < alias_probs[chosen_idx]) {
            samples[samp_idx * samp_int] = chosen_idx;
        }
        else {
            samples[samp_idx * samp_int] = aliases[chosen_idx];
        }
    }
}
