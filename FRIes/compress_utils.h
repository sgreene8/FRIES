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

//#define USE_MPI
#ifdef USE_MPI
#include "mpi.h"
#endif

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
                     size_t count, unsigned int *n_samp, double *global_norm,
                     double *fmax);

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
 values: vector on which to perform compression. Elements can be negative
 
 */


#endif /* compress_utils_h */
