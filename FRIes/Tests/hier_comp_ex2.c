#include <stdio.h>
#include <stdlib.h>
#include "dc.h"
#include <time.h>
#include "compress_utils.h"

#define n_wt 100
#define n_sub 10

int main(int argc, const char * argv[]) {
    int n_procs = 1;
    int proc_rank = 0;
#ifdef USE_MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
#endif
    
    double orig_weights[n_wt];
    unsigned int counts[n_wt];
    double orig_sub[n_wt][n_sub];
    double accum_mean[n_wt][n_sub];
    
    mt_struct *rngen_ptr = get_mt_parameter_id_st(32, 607, proc_rank, (unsigned int) time(NULL));
    sgenrand_mt((uint32_t) time(NULL), rngen_ptr);
    
    size_t i, j, max_i, max_j;
    double tot_weight;
    for (i = 0; i < n_wt; i += 2) {
        orig_weights[i] = genrand_mt(rngen_ptr) / MT_MAX * 10;
        counts[i] = 0;
        tot_weight = 0;
        for (j = 0; j < n_sub; j++) {
            orig_sub[i][j] = genrand_mt(rngen_ptr) / MT_MAX;
            tot_weight += orig_sub[i][j];
        }
        for (j = 0; j < n_sub; j++) {
            orig_sub[i][j] /= tot_weight;
            accum_mean[i][j] = 0;
        }
    }
    
    for (i = 1; i < n_wt; i += 2) {
        orig_weights[i] = genrand_mt(rngen_ptr) / MT_MAX * 10;
        counts[i] = (genrand_mt(rngen_ptr) % n_sub) + 1;
        for (j = 0; j < counts[i]; j++) {
            accum_mean[i][j] = 0;
        }
    }
    
    int keep_idx[n_wt][n_sub];
    
    for (i = 0; i < n_wt; i++) {
        if (counts[i] > 0) {
            keep_idx[i][0] = 0;
        }
        else {
            for (j = 0; j < n_sub; j++) {
                keep_idx[i][j] = 0;
            }
        }
    }
    
    unsigned int n_samp = 200;
    double cmp_vec[n_samp];
    size_t cmp_idx[n_samp][2];
    double wt_remain[n_wt];
    double loc_norms[n_procs];
    
    double rn_sys = 0.0;
    unsigned int n_iter = 1000;
    unsigned int iter;
    
    char path[100];
    sprintf(path, "max_dev_%d.txt", proc_rank);
    FILE *dev_file = fopen(path, "w");
    double max_dev, diff;
    
    for (iter = 0; iter < n_iter; iter++) {
        loc_norms[proc_rank] = find_keep_sub(orig_weights, counts, n_sub, orig_sub, keep_idx, n_wt, &n_samp, wt_remain);
        if (proc_rank == 0) {
            rn_sys = genrand_mt(rngen_ptr) / MT_MAX;
        }
#ifdef USE_MPI
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, loc_norms, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Bcast(&rn_sys, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
        n_samp = sys_sub(orig_weights, counts, n_sub, orig_sub, keep_idx, n_wt, n_samp, wt_remain, loc_norms, rn_sys, cmp_vec, cmp_idx);
        
        for (i = 0; i < n_samp; i++) {
            accum_mean[cmp_idx[i][0]][cmp_idx[i][1]] += cmp_vec[i];
        }
        
        max_dev = 0;
        max_i = 0;
        max_j = 0;
        for (i = 0; i < n_wt; i++) {
            if (counts[i] > 0) {
                for (j = 0; j < counts[i]; j++) {
                    diff = fabs(accum_mean[i][j] / (iter + 1) - orig_weights[i] / counts[i]);
                    if (diff > max_dev) {
                        max_dev = diff;
                        max_i = i;
                        max_j = j;
                    }
                }
            }
            else {
                for (j = 0; j < n_sub; j++) {
                    diff = fabs(accum_mean[i][j] / (iter + 1) - orig_weights[i] * orig_sub[i][j]);
                    if (diff > max_dev) {
                        max_dev = diff;
                        max_i = i;
                        max_j = j;
                    }
                }
            }
        }
        fprintf(dev_file, "%lf\n", max_dev);
    }
    fclose(dev_file);
    
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}

