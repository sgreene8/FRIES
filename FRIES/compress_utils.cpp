/*! \file
 *
 * \brief Utilities for compressing vectors stochastically using the FRI
 * framework
 */

#include "compress_utils.hpp"


int round_binomially(double p, unsigned int n, mt_struct *mt_ptr) {
    int flr = floor(p);
    double prob = p - flr;
    int ret_val = flr * n;
    for (unsigned int i = 0; i < n; i++) {
        ret_val += (genrand_mt(mt_ptr) / (1. + UINT32_MAX)) < prob;
    }
    return ret_val;
}

double find_preserve(double *values, size_t *srt_idx, std::vector<bool> &keep_idx,
                     size_t count, unsigned int *n_samp, double *global_norm) {
    double loc_one_norm = 0;
    double glob_one_norm = 0;
    size_t heap_count = count;
    for (size_t det_idx = 0; det_idx < count; det_idx++) {
        loc_one_norm += fabs(values[det_idx]);
    }
    int proc_rank = 0;
    int n_procs = 1;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
#endif
    
    auto val_compare = [values](size_t i, size_t j){return fabs(values[i]) < fabs(values[j]); };
    std::make_heap(srt_idx, srt_idx + heap_count, val_compare);
    int loc_sampled, glob_sampled = 1;
    int keep_going = 1;
    
    double el_magn = 0;
    size_t max_idx;
    *global_norm = sum_mpi(loc_one_norm, proc_rank, n_procs);
    while (glob_sampled > 0) {
        glob_one_norm = sum_mpi(loc_one_norm, proc_rank, n_procs);
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
                    std::pop_heap(srt_idx, srt_idx + heap_count + 1, val_compare);
                }
                else {
                    keep_going = 0;
                }
            }
            else{
                keep_going = 0;
            }
        }
        glob_sampled = sum_mpi(loc_sampled, proc_rank, n_procs);
        (*n_samp) -= glob_sampled;
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
#ifdef USE_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
#endif
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
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
#endif
    
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
    int n_procs = 1;
    int proc_rank = 0;
    double rn_sys = rand_num;
#ifdef USE_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Bcast(&rn_sys, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
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
#ifdef USE_MPI
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, loc_norms, 1, MPI_DOUBLE, MPI_COMM_WORLD);
#endif
}

void sys_comp_nonuni(double *vec_vals, size_t vec_len, double *loc_norms,
                     unsigned int n_samp, std::vector<bool> &keep_exact,
                     double *probs, size_t n_probs, mt_struct *rn_gen) {
    int n_procs = 1;
    int proc_rank = 0;
#ifdef USE_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
#endif
    double rn_sys = 0;
    if (proc_rank == 0) {
        unsigned int aliases [n_probs];
        double alias_probs [n_probs];
        setup_alias(probs, aliases, alias_probs, n_probs);
        uint8_t sample;
        sample_alias(aliases, alias_probs, n_probs, &sample, 1, 1, rn_gen);
        rn_sys = genrand_mt(rn_gen) / (1. + UINT32_MAX) / n_probs + (double) sample / n_probs;
    }
#ifdef USE_MPI
    MPI_Bcast(&rn_sys, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
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
                if (rn_idx == (size_t)right_rn_idx) {
                    pdf_frac = right_rn_idx - left_rn_idx;
                }
                else {
                    pdf_frac = (size_t)(left_rn_idx + 1) - left_rn_idx;
                }
                double pdf_integral = pdf_frac * probs[rn_idx % n_probs];
                for (rn_idx++; rn_idx < right_rn_idx - 1; rn_idx++) {
                    pdf_integral += probs[rn_idx % n_probs];
                }
                if (rn_idx != (size_t)right_rn_idx) {
                    pdf_frac = right_rn_idx - (size_t)right_rn_idx;
                    pdf_integral += probs[rn_idx % n_probs];
                }
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
#ifdef USE_MPI
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, loc_norms, 1, MPI_DOUBLE, MPI_COMM_WORLD);
#endif
}


void sys_obs(double *vec_vals, size_t vec_len, double *loc_norms, unsigned int n_samp,
             std::vector<bool> &keep_exact, std::function<double(size_t)> obs,
             double *obs_vals, size_t num_rns) {
    int n_procs = 1;
    int proc_rank = 0;
#ifdef USE_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
#endif
    double glob_norm = 0;
    for (int proc_idx = 0; proc_idx < n_procs; proc_idx++) {
        glob_norm += loc_norms[proc_idx];
    }
    
    for (size_t rn_idx = 0; rn_idx < num_rns; rn_idx++) {
        obs_vals[rn_idx] = 0;
    }
    
    double mag_before = 0;
    for (int proc_idx = 0; proc_idx < proc_rank; proc_idx++) {
        mag_before += loc_norms[proc_idx];
    }
    
    double left_rn_idx = mag_before / (glob_norm / n_samp / num_rns);
    
    for (size_t el_idx = 0; el_idx < vec_len; el_idx++) {
        double el_obs = obs(el_idx);
        double tmp_val = fabs(vec_vals[el_idx]);
        if (keep_exact[el_idx]) {
            for (size_t rn_idx = 0; rn_idx < num_rns; rn_idx++) {
                obs_vals[rn_idx] += el_obs * tmp_val * tmp_val;
            }
        }
        else {
            mag_before += tmp_val;
            double right_rn_idx = mag_before / (glob_norm / n_samp / num_rns);
            size_t rn_idx = left_rn_idx;
            if (fabs(rn_idx - left_rn_idx) != 0) {
                rn_idx++;
            }
            if (fabs(right_rn_idx - n_samp * num_rns) < 1e-8) { // correct for floating-point error
                right_rn_idx = n_samp * num_rns - 0.5;
            }
            for (; rn_idx < right_rn_idx; rn_idx++) {
                obs_vals[rn_idx % num_rns] += el_obs * glob_norm / n_samp * glob_norm / n_samp;
            }
            left_rn_idx = right_rn_idx;
        }
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
#ifdef USE_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Bcast(&rn_sys, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
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
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Bcast(&rand_num, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    unsigned int tmp_nsamp = n_samp;
    double loc_norms[n_procs];
    if (keep_idx.cols() != sub_weights.cols()) {
        fprintf(stderr, "Error in comp_sub: column dimension of sub_weights does not equal column dimension of keep_idx.\n");
        return 0;
    }
    
    loc_norms[proc_rank] = find_keep_sub(values, n_div, sub_weights, keep_idx, sub_sizes, count, &tmp_nsamp, wt_remain);
#ifdef USE_MPI
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, loc_norms, 1, MPI_DOUBLE, MPI_COMM_WORLD);
#endif
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
                  mt_struct *mt_ptr) {
    uint8_t chosen_idx;
    for (unsigned int samp_idx = 0; samp_idx < n_samp; samp_idx++) {
        chosen_idx = genrand_mt(mt_ptr) / (1. + UINT32_MAX) * n_states;
        if (genrand_mt(mt_ptr) / (1. + UINT32_MAX) < alias_probs[chosen_idx]) {
            samples[samp_idx * samp_int] = chosen_idx;
        }
        else {
            samples[samp_idx * samp_int] = aliases[chosen_idx];
        }
    }
}
