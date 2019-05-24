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
        ret_val += (genrand_mt(mt_ptr) / MT_MAX) < prob;
    }
    return ret_val;
}

double find_preserve(double *values, size_t *srt_idx, int *keep_idx,
                     size_t count, unsigned int *n_samp, double *global_norm,
                     double *fmax) {
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
    if (fabs(glob_one_norm) < 1e-9) {
        *n_samp = 0;
    }
    else {
        for (det_idx = 0; det_idx < count; det_idx++) {
            if (!keep_idx[det_idx]) {
                loc_one_norm += fabs(values[det_idx]);
            }
        }
    }
    *fmax = el_magn;
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
