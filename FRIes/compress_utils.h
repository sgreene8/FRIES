//
//  compress_utils.h
//  FRIes
//
//  Created by Samuel Greene on 4/13/19.
//  Copyright © 2019 Samuel Greene. All rights reserved.
//

#ifndef compress_utils_h
#define compress_utils_h

#include <stdio.h>
#include "dc.h"
#include <math.h>
#include "heap.h"
#include "mpi_switch.h"
#include "det_store.h"

/* Round p to integer b such that
 b ~ binomial(n, p - floor(p)) + floor(p) * n
 
 Parameters
 ----------
 p: non-integer number to be rounded
 n: Number of rn's to sample
 mt_ptr: Address to MT state object to use for RN generation
 
 Returns
 -------
 integer result
 
 */
int round_binomially(double p, unsigned int n, mt_struct *mt_ptr);


/* Identify the greatest-magnitude elements of a vector to be preserved
 in a compression
 
 Parameters
 ----------
 values: vector of elements (can be + or -); not modified in this subroutine
 srt_idx: an array of indices that will be used to build the heap, must already
    contain integers from 0 to count - 1 in any order
 keep_idx: upon return, contains 1's at each position of values that should be
    preserved exactly. all elements should be zeroed before calling
 count: number of elements in values array
 n_samp: pointer to desired number of nonzero elements after compression;
    upon return, points to remaining number available for systematic resampling
 global_norm: upon return, contains the norm of the whole vector, including
    preserved elements
 
 Returns
 -------
 sum of magnitudes of elements that are not preserved exactly
 */
double find_preserve(double *values, size_t *srt_idx, int *keep_idx,
                     size_t count, unsigned int *n_samp, double *global_norm);

/* Sum a variable across all processors and store the sum in the global ptr
 
 Parameters
 ----------
 local: local value to be summed
 global: pointer to where result should be stored
 my_rank: rank of calling process
 n_procs: total number of processors to sum over
 */
void sum_mpi_d(double local, double *global, int my_rank, int n_procs);


/* Same as sum_mpi_d, except for integers
 */
void sum_mpi_i(int local, int *global, int my_rank, int n_procs);

/* Initialize the parameters needed to perform systematic sampling on a vector of weights
 
 Parameters
 ----------
 norms: array of one-norms of the portion of the vector stored on each processor
 rn: pointer to a random number on [0,1). Upon return, will be set to the position of the random sampler (the "X") within this portion of the vector.
 n_samp: desired number of elements to select in systematic sampling
 */
double seed_sys(double *norms, double *rn, unsigned int n_samp);


/*
 Identify elements of a vector to preserve in a systematic FRI compression scheme.
 Vector elements are subdivided into sub-weights.
 
 Parameters
 ----------
 values: vector on which to perform compression. Elements must be positive.
 n_div: number of uniform intervals into which vector elements are divided. If =0,
    vector is divided nonuniformly at this position
 n_sub: length of 2nd dimension of sub_wts and keep_idx arrays
 sub_wts: 2-d array of sub-weights for vector elements divided nonuniformly;
            each nonzero row must sum to 1
 keep_idx: 2-d array that, upon return, contains 1's at all positions to be
            preserved exactly. Relevant indices should be zeroed before calling.
 count: length of values array
 n_samp: pointer to desired number of nonzero elements after compression;
    upon return, points to remaining number available for systematic resampling
 wt_remain: array that, upon return, contains the remaining weight to be sampled
    at each position
 
 Returns
 -------
 sum of magnitudes of elements that are not preserved exactly
 */
double find_keep_sub(double *values, unsigned int *n_div, size_t n_sub,
                     double (*sub_weights)[n_sub], int (*keep_idx)[n_sub],
                     size_t count, unsigned int *n_samp, double *wt_remain);


/*
 Systematically compress a vector
 
 Parameters
 ----------
 vec_vals: values of elements in the vector
 vec_len: number of elements in the vector
 loc_norms: one-norm of the segments of the solution vector on each processor
            updated upon return
 n_samp: number of samples to sample using systematic sampling
 keep_exact: array indicating which elements should be preserved exactly; upon
             return, 1's indicate elements that should be removed from solution
             vector
 rand_num: random number on [0,1)
 */
void sys_comp(double *vec_vals, size_t vec_len, double *loc_norms,
                unsigned int n_samp, int *keep_exact, double rand_num);


/*
 Adjust energy shift according to eq 17 in Booth et al. (2009)
 
 Parameters
 ----------
 shift: pointer to current energy shift; updated upon return
 one_norm: current one-norm of solution vector
 last_norm: ptr to previous one-norm of solution vector; updated upon return
 target_norm: one-norm above which the energy shift should be adjusted
 damp_factor: prefactor for log in eq 17 (\zeta / A \delta \tau)
 */
void adjust_shift(double *shift, double one_norm, double *last_norm,
                  double target_norm, double damp_factor);

#endif /* compress_utils_h */
