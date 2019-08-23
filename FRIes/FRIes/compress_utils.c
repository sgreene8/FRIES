//
//  compress_utils.c
//  FRIes
//
//  Created by Samuel Greene on 4/13/19.
//  Copyright © 2019 Samuel Greene. All rights reserved.
//

#include "compress_utils.h"


int round_binomially(double p, unsigned int n, mt_struct *mt_ptr) {
    int flr = floor(p);
    double prob = p - flr;
    int ret_val = flr * n;
    unsigned int i;
    for (i = 0; i < n; i++) {
        ret_val += (genrand_mt(mt_ptr) / (1. + UINT32_MAX)) < prob;
    }
    return ret_val;
}

double find_preserve(double *values, size_t *srt_idx, int *keep_idx,
                     size_t count, unsigned int *n_samp, double *global_norm) {
    double loc_one_norm = 0;
    double glob_one_norm = 0;
    size_t det_idx;
    size_t heap_count = count;
    for (det_idx = 0; det_idx < count; det_idx++) {
        loc_one_norm += fabs(values[det_idx]);
    }
    int proc_rank = 0;
    int n_procs = 1;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
#endif
    
    heapify(values, srt_idx, heap_count);
    int loc_sampled, glob_sampled = 1;
    int keep_going = 1;
    
    double el_magn = 0;
    size_t max_idx;
    sum_mpi_d(loc_one_norm, global_norm, proc_rank, n_procs);
    while (glob_sampled > 0) {
        sum_mpi_d(loc_one_norm, &glob_one_norm, proc_rank, n_procs);
        loc_sampled = 0;
        while (keep_going) {
            max_idx = srt_idx[0];
            el_magn = fabs(values[max_idx]);
            if (heap_count > 0 && el_magn >= glob_one_norm / (*n_samp - loc_sampled)) {
                keep_idx[max_idx] = 1;
                loc_sampled++;
                loc_one_norm -= el_magn;
                glob_one_norm -= el_magn;
                
                heap_count--;
                if (heap_count) {
                    srt_idx[0] = srt_idx[heap_count];
                    srt_idx[heap_count] = max_idx;
                    sift_down(values, srt_idx, 0, heap_count - 1);
                }
                else {
                    keep_going = 0;
                }
            }
            else{
                keep_going = 0;
            }
        }
        sum_mpi_i(loc_sampled, &glob_sampled, proc_rank, n_procs);
        (*n_samp) -= glob_sampled;
        keep_going = 1;
    }
    loc_one_norm = 0;
    if (glob_one_norm < 1e-9) {
        *n_samp = 0;
    }
    else {
        for (det_idx = 0; det_idx < count; det_idx++) {
            if (!keep_idx[det_idx]) {
                loc_one_norm += fabs(values[det_idx]);
            }
        }
    }
    return loc_one_norm;
}


void sum_mpi_d(double local, double *global, int my_rank, int n_procs) {
    int proc_idx;
    double rec_vals[n_procs];
    rec_vals[my_rank] = local;
#ifdef USE_MPI
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, rec_vals, 1, MPI_DOUBLE, MPI_COMM_WORLD);
#endif
    *global = 0;
    for (proc_idx = 0; proc_idx < n_procs; proc_idx++) {
        (*global) += rec_vals[proc_idx];
    }
}


void sum_mpi_i(int local, int *global, int my_rank, int n_procs) {
    int proc_idx;
    int rec_vals[n_procs];
    rec_vals[my_rank] = local;
#ifdef USE_MPI
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_INT, rec_vals, 1, MPI_INT, MPI_COMM_WORLD);
#endif
    *global = 0;
    for (proc_idx = 0; proc_idx < n_procs; proc_idx++) {
        (*global) += rec_vals[proc_idx];
    }
}

double seed_sys(double *norms, double *rn, unsigned int n_samp) {
    double lbound = 0;
    int proc_idx = 0;
    int n_procs = 1;
    int my_rank = 0;
#ifdef USE_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
#endif
    double global_norm;
    for (proc_idx = 0; proc_idx < my_rank; proc_idx++) {
        lbound += norms[proc_idx];
    }
    global_norm = lbound;
    for (; proc_idx < n_procs; proc_idx++) {
        global_norm += norms[proc_idx];
    }
    *rn *= global_norm / n_samp;
    *rn += global_norm / n_samp * (int)(lbound * n_samp / global_norm);
    if (*rn < lbound) {
        *rn += global_norm / n_samp;
    }
    return lbound;
}


double find_keep_sub(double *values, unsigned int *n_div, size_t n_sub,
                     double (*sub_weights)[n_sub], int (*keep_idx)[n_sub],
                     size_t count, unsigned int *n_samp, double *wt_remain) {
    double loc_one_norm = 0;
    double glob_one_norm = 0;
    size_t det_idx, sub_idx;
    for (det_idx = 0; det_idx < count; det_idx++) {
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
    double el_magn, sub_magn, keep_thresh, sub_remain;
    while (glob_sampled > 0) {
        sum_mpi_d(loc_one_norm, &glob_one_norm, proc_rank, n_procs);
        if (glob_one_norm < 0) {
            break;
        }
        loc_sampled = 0;
        for (det_idx = 0; det_idx < count; det_idx++) {
            el_magn = values[det_idx];
            keep_thresh = glob_one_norm / (*n_samp - loc_sampled);
            if (el_magn >= keep_thresh) {
                if (n_div[det_idx] > 0) {
                    if (el_magn / n_div[det_idx] >= keep_thresh && !keep_idx[det_idx][0]) {
                        keep_idx[det_idx][0] = 1;
                        wt_remain[det_idx] = 0;
                        loc_sampled += n_div[det_idx];
                        loc_one_norm -= el_magn;
                        glob_one_norm -= el_magn;
                        if (glob_one_norm < 0) {
                            break;
                        }
                    }
                }
                else {
                    sub_remain = 0;
                    for (sub_idx = 0; sub_idx < n_sub; sub_idx++) {
                        if (!keep_idx[det_idx][sub_idx]) {
                            sub_magn = el_magn * sub_weights[det_idx][sub_idx];
                            if (sub_magn >= keep_thresh && fabs(sub_magn) > 1e-10) {
                                keep_idx[det_idx][sub_idx] = 1;
                                loc_sampled++;
                                loc_one_norm -= sub_magn;
                                glob_one_norm -= sub_magn;
                                if (glob_one_norm < 0) {
                                    wt_remain[det_idx] = 0;
                                    break;
                                }
                                keep_thresh = glob_one_norm / (*n_samp - loc_sampled);
                            }
                            else {
                                sub_remain += sub_magn;
                            }
                        }
                    }
                    wt_remain[det_idx] = sub_remain;
                }
            }
        }
        sum_mpi_i(loc_sampled, &glob_sampled, proc_rank, n_procs);
        (*n_samp) -= glob_sampled;
    }
    loc_one_norm = 0;
    if (glob_one_norm < 1e-7) {
        *n_samp = 0;
    }
    else {
        for (det_idx = 0; det_idx < count; det_idx++) {
            loc_one_norm += wt_remain[det_idx];
        }
    }
    return loc_one_norm;
}

void sys_comp(double *vec_vals, size_t vec_len, double *loc_norms,
              unsigned int n_samp, int *keep_exact, double rand_num) {
    int n_procs = 1;
    int proc_rank = 0;
    unsigned int proc_idx;
    double rn_sys = rand_num;
#ifdef USE_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Bcast(&rn_sys, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    double tmp_glob_norm = 0;
    for (proc_idx = 0; proc_idx < n_procs; proc_idx++) {
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
    size_t det_idx;
    double tmp_val;
    for (det_idx = 0; det_idx < vec_len; det_idx++) {
        tmp_val = vec_vals[det_idx];
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

size_t sys_sub(double *values, unsigned int *n_div, size_t n_sub,
               double (*sub_weights)[n_sub], int (*keep_idx)[n_sub],
               size_t count, unsigned int n_samp, double *wt_remain,
               double *loc_norms, double rand_num, double *new_vals,
               size_t (*new_idx)[2]) {
    int n_procs = 1;
    int proc_rank = 0;
    unsigned int proc_idx;
    double rn_sys = rand_num;
#ifdef USE_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Bcast(&rn_sys, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    double tmp_glob_norm = 0;
    for (proc_idx = 0; proc_idx < n_procs; proc_idx++) {
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
    size_t wt_idx, sub_idx;
    double tmp_val;
    size_t num_new = 0;
    double sub_lbound;
    for (wt_idx = 0; wt_idx < count; wt_idx++) {
        tmp_val = values[wt_idx];
        lbound += wt_remain[wt_idx];
        if (n_div[wt_idx] > 0) {
            if (keep_idx[wt_idx][0]) {
                keep_idx[wt_idx][0] = 0;
                for (sub_idx = 0; sub_idx < n_div[wt_idx]; sub_idx++) {
                    new_vals[num_new] = tmp_val / n_div[wt_idx];
                    new_idx[num_new][0] = wt_idx;
                    new_idx[num_new][1] = sub_idx;
                    num_new++;
                }
                loc_norms[proc_rank] += tmp_val;
            }
            else {
                while (rn_sys < lbound) {
                    sub_idx = (lbound - rn_sys) * n_div[wt_idx] / tmp_val;
                    new_vals[num_new] = tmp_glob_norm / n_samp;
                    new_idx[num_new][0] = wt_idx;
                    new_idx[num_new][1] = sub_idx;
                    num_new++;
                    rn_sys += tmp_glob_norm / n_samp;
                    loc_norms[proc_rank] += tmp_glob_norm / n_samp;
                }
            }
        }
        else if (wt_remain[wt_idx] < tmp_val || rn_sys < lbound) {
            loc_norms[proc_rank] += (tmp_val - wt_remain[wt_idx]);
            sub_lbound = lbound - wt_remain[wt_idx];
            for (sub_idx = 0; sub_idx < n_sub; sub_idx++) {
                if (keep_idx[wt_idx][sub_idx]) {
                    keep_idx[wt_idx][sub_idx] = 0;
                    new_vals[num_new] = tmp_val * sub_weights[wt_idx][sub_idx];
                    new_idx[num_new][0] = wt_idx;
                    new_idx[num_new][1] = sub_idx;
                    num_new++;
                }
                else {
                    sub_lbound += tmp_val * sub_weights[wt_idx][sub_idx];
                    if (rn_sys < sub_lbound) {
                        new_vals[num_new] = tmp_glob_norm / n_samp;
                        new_idx[num_new][0] = wt_idx;
                        new_idx[num_new][1] = sub_idx;
                        num_new++;
                        loc_norms[proc_rank] += tmp_glob_norm / n_samp;
                        rn_sys += tmp_glob_norm / n_samp;
                    }
                }
            }
        }
    }
    return num_new;
}


size_t comp_sub(double *values, size_t count, unsigned int *n_div, size_t n_sub,
                double (*sub_weights)[n_sub], int (*keep_idx)[n_sub],
                unsigned int n_samp, double *wt_remain, double rand_num,
                double *new_vals, size_t (*new_idx)[2]) {
    int proc_rank = 0;
    int n_procs = 1;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Bcast(&rand_num, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    unsigned int tmp_nsamp = n_samp;
    double loc_norms[n_procs];
    
    loc_norms[proc_rank] = find_keep_sub(values, n_div, n_sub, sub_weights, keep_idx, count, &tmp_nsamp, wt_remain);
#ifdef USE_MPI
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, loc_norms, 1, MPI_DOUBLE, MPI_COMM_WORLD);
#endif
    return sys_sub(values, n_div, n_sub, sub_weights, keep_idx, count, tmp_nsamp, wt_remain, loc_norms, rand_num, new_vals, new_idx);
}
