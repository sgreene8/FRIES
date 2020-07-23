/*! \file
 *
 * \brief Utilities for compressing vectors stochastically using the FRI
 * framework
 */

#ifndef compress_utils_h
#define compress_utils_h

#include <cstdio>
#include <cstdint>
#include <climits>
#include <cmath>
#include <algorithm>
#include <FRIES/mpi_switch.h>
#include <FRIES/det_store.h>
#include <FRIES/Ext_Libs/dcmt/dc.h>
#include <FRIES/ndarr.hpp>
#include <functional>
#include <iostream>

/*! \brief Round a non-integral number binomially.
 *
 * Given a non-integral input p and a positive-integral input n, the result r of
 * this operation is distributed according to:
 * \f[
 * r \sim \text{binomial}(n, p - \text{floor}(p)) + \text{floor}(p) * n
 * \f]
 * \param [in] p        The non-integral parameter p of the rounding operation
 * \param [in] n        The positive integral parameter n of the rounding oper.
 * \param [in] mt_ptr   Address to MT state object to use for RN generation
 * \return Integer result r of the operation
 */
int round_binomially(double p, unsigned int n, mt_struct *mt_ptr);


/*! \brief Identify the greatest-magnitude elements of a vector
 *
 * The greatest-magnitude elements of the vector are identified according to the
 * rule in the FRI paper and preserved exactly in the compression
 *
 * \param [in] values   Vector of elements (can be + or -); not modified in this
 *                      subroutine (length \p count)
 * \param [in] srt_idx  An array of indices that will be used to build the heap,
 *                      must be initialized with integers from 0 to count - 1 in
 *                      any order (length \p count)
 * \param [out] keep_idx Contains 1's at each position of values designated to
 *                      be preserved exactly. Upon input, all elements must be 0
 *                      (length \p count)
 * \param [in] count    Number of elements in the vector being compressed
 * \param [in,out] n_samp Pointer to desired number of nonzero elements after
 *                      compression. Upon return, points to remaining number
 *                      available for systematic resampling
 * \param [out] global_norm The norm of the whole vector, including
 *                      preserved elements
 * \return Sum of magnitudes of elements that are not preserved exactly
 */
double find_preserve(double *values, size_t *srt_idx, std::vector<bool> &keep_idx,
                     size_t count, unsigned int *n_samp, double *global_norm);


/*! \brief Systematic resampling of vector elements
 *
 * \param [in, out] vec_vals Elements in the vector (can be negative) before
 *                      and after compression (length \p vec_len)
 * \param [in] vec_len  Number of elements in the vector
 * \param [in, out] loc_norms Sum of magnitudes of elements on each MPI process
 *                      before and after compression
 * \param [in] n_samp   Number of samples in systematic resampling
 * \param [in, out] keep_exact Array indicating elements to be preserved exactly
 *                      in compression; upon return, 1's indicate elements
 *                      zeroed in the compression
 * \param [in] rand_num A random number chosen uniformly on [0, 1). Only the
 *                      argument from the 0th MPI process is used.
 */
void sys_comp(double *vec_vals, size_t vec_len, double *loc_norms,
              unsigned int n_samp, std::vector<bool> &keep_exact, double rand_num);

/*! \brief Systematic resampling of vector elements only on this process. Does not use MPI
 *
 * \param [in, out] vec_vals Elements in the vector (can be negative) before
 *                      and after compression (length \p vec_len)
 * \param [in] vec_len  Number of elements in the vector
 * \param [in] seg_norm  Sum of magnitudes of elements in this segment not preserved exactly
 * \param [in] sampl_val    Magnitude to assign to elements selected
 * \param [in] n_samp   Number of samples in systematic resampling (just for this segment)
 * \param [in, out] keep_exact Array indicating elements to be preserved exactly
 *                      in compression; upon return, 1's indicate elements
 *                      zeroed in the compression
 * \param [in] rand_num A random number chosen uniformly on [0, 1)
 */
void sys_comp_serial(double *vec_vals, size_t vec_len, double seg_norm, double sampl_val,
                     uint32_t n_samp, std::vector<bool> &keep_exact, double rand_num);


/*! \brief Pivotal resampling of vector elements only on this process. Does not use MPI
 *
 * \param [in, out] vec_vals Elements in the vector (can be negative) before
 *                      and after compression (length \p vec_len)
 * \param [in] vec_len  Number of elements in the vector
 * \param [in] seg_norm  Sum of magnitudes of elements in this segment not preserved exactly
 * \param [in] sampl_val    Magnitude to assign to elements selected
 * \param [in] n_samp   Number of samples in systematic resampling (just for this segment)
 * \param [in, out] keep_exact Array indicating elements to be preserved exactly
 *                      in compression; upon return, 1's indicate elements
 *                      zeroed in the compression
 * \param [in] rngen_ptr   A pointer to a mt_struct object to use for random number generation
 */
void piv_comp_serial(double *vec_vals, size_t vec_len, double seg_norm, double sampl_val,
                     uint32_t n_samp, std::vector<bool> &keep_exact, mt_struct *rngen_ptr);



/*! \brief Calculate a budget for compressing vectors on each MPI process independently using systematic resampling
 *
 * \param [in] loc_norms    Sum of magnitudes of elements on each MPI process
 * \param [in] n_samp       Total number of elements (across all processes) to select in sampling
 * \param [in] rand_num A random number chosen uniformly on [0, 1). Only the
 *                      argument from the 0th MPI process is used.
 * \return Number of elements to sample on this process
 */
 uint32_t sys_budget(double *loc_norms, uint32_t n_samp, double rand_num);


/*! \brief If needed, adjust elements in a vector prior to resampling to ensure that each element can be selected
 * only 0 or 1 times
 *
 * \param [in, out] vec_vals Elements in the vector (can be negative) to be compressed
 * \param [in] vec_len  Number of elements in this segment of the vector
 * \param [in] n_samp_loc   The number of elements to be selected from this segment only
 * \param [in] exp_nsamp_loc The expected value of \p n_samp_loc
 * \param [in] n_samp_tot   The total number of elements that will be selected from all segments
 * \param [in] tot_norm     The total one-norm of all vector segmemts
 * \param [in] keep_exact Array indicating elements to be preserved exactly
 *                      in compression
 * \return One-norm of the segment after adjusting the probabilities
 */
double adjust_probs(double *vec_vals, size_t vec_len, uint32_t n_samp_loc,
                  double exp_nsamp_loc, uint32_t n_samp_tot, double tot_norm,
                  std::vector<bool> &keep_exact);


/*! \brief Systematic resampling of vector elements using an arbitrary (not necessarily uniform) probability distribution
 *
 * \param [in, out] vec_vals Elements in the vector (can be negative) before
 *                      and after compression (length \p vec_len)
 * \param [in] vec_len  Number of elements in the vector
 * \param [in, out] loc_norms Sum of magnitudes of elements on each MPI process
 *                      before and after compression
 * \param [in] n_samp   Number of samples in systematic resampling
 * \param [in, out] keep_exact Array indicating elements to be preserved exactly
 *                      in compression; upon return, 1's indicate elements
 *                      zeroed in the compression
 * \param [in] probs    Probability distribution used to generate the random number used for compression
 * \param [in] n_probs  Number of elements in \p probs
 * \param [in] rand_num A random number chosen uniformly on [0, 1). Only the
 *                      argument from the 0th MPI process is used.
 */
void sys_comp_nonuni(double *vec_vals, size_t vec_len, double *loc_norms,
                     unsigned int n_samp, std::vector<bool> &keep_exact,
                     double *probs, size_t n_probs, double rand_num);


/*! \brief Calculate the value of some diagonal observable that would be obtained if different values were used for
 *  the uniformly sampled random number inputted to systematic resampling
 *
 *  \param [in] vec_vals Elements in the vector (can be negative) to be compressed (length \p vec_len)
 * \param [in] vec_len  Number of elements in the vector
 * \param [in] loc_norms Sum of magnitudes of elements on each MPI process not preserved exactly
 * \param [in] n_samp   Number of samples in systematic resampling
 * \param [in] keep_exact Array indicating elements to be preserved exactly
 *                      in compression
 * \param [in] obs      Function that takes the index of an element in \p vec_vals and adds to an array the values of the observables
 * \param [out] obs_vals    The values of the observables for each of the possible random numbers that
 *              can be used in systematic compression
 */
void sys_obs(double *vec_vals, size_t vec_len, double *loc_norms, unsigned int n_samp,
             std::vector<bool> &keep_exact, std::function<void(size_t, double *)> obs,
             Matrix<double> &obs_vals);


/*! \brief Sum a variable across all MPI processes
 *
 * \param [in] local    local value to be summed
 * \param [in] my_rank  Rank of the local processor
 * \param [in] n_procs  Total number of MPI processes
 * \return The calculated sum
 */
inline double sum_mpi(double local, int my_rank, int n_procs)  {
    double rec_vals[n_procs];
    rec_vals[my_rank] = local;
#ifdef USE_MPI
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, rec_vals, 1, MPI_DOUBLE, MPI_COMM_WORLD);
#endif
    double global = 0;
    for (int proc_idx = 0; proc_idx < n_procs; proc_idx++) {
        global += rec_vals[proc_idx];
    }
    return global;
}


/*! \brief Sum a variable across all MPI processes
 *
 * \param [in] local    local value to be summed
 * \param [in] my_rank  Rank of the local processor
 * \param [in] n_procs  Total number of MPI processes
 * \return The calculated sum
 */
inline int sum_mpi(int local, int my_rank, int n_procs) {
    int rec_vals[n_procs];
    rec_vals[my_rank] = local;
#ifdef USE_MPI
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_INT, rec_vals, 1, MPI_INT, MPI_COMM_WORLD);
#endif
    int global = 0;
    for (int proc_idx = 0; proc_idx < n_procs; proc_idx++) {
        global += rec_vals[proc_idx];
    }
    return global;
}


/*! \brief Sum a variable across all MPI processes
 *
 * \param [in] local    local value to be summed
 * \param [in] my_rank  Rank of the local processor
 * \param [in] n_procs  Total number of MPI processes
 * \return The calculated sum
 */
inline uint64_t sum_mpi(uint64_t local, int my_rank, int n_procs) {
    uint64_t rec_vals[n_procs];
    rec_vals[my_rank] = local;
#ifdef USE_MPI
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_UINT64_T, rec_vals, 1, MPI_UINT64_T, MPI_COMM_WORLD);
#endif
    uint64_t global = 0;
    for (int proc_idx = 0; proc_idx < n_procs; proc_idx++) {
        global += rec_vals[proc_idx];
    }
    return global;
}


/*! \brief Set-up for performing systematic compression across many MPI processes
 *
 * Calculates the position of the first random sample on each MPI process
 *
 * \param [in] norms    Array of one-norms of the portions of the vector stored
 *                      on each processor (length \p n_samp)
 * \param [in, out] rn  Pointer to a random number generated on [0,1). Upon
 *                      return, will be set to the position of the random sample
 *                      (the "X" in the paper) within this portion of the vector
 * \param [in] n_samp   Desired number of elements to select randomly
 * \return Sum of the one-norms of the portions of the vector stored on MPI
 * processes with ranks less than the current one
 */
double seed_sys(double *norms, double *rn, unsigned int n_samp);


/*! \brief Identify elements to preserve exactly according to the FRI rule when
 * vector elements are subdivided into sub-weights.
 *
 * \param [in] values   Magnitudes of elements of vector to be compressed
 *                      (length \p count)
 * \param [in] n_div    Number of uniform intervals into which each element is
 *                      divided. If =0, this element is divided nonuniformly
 *                      according to sub_wts at this position (length \p count)
 * \param [in] sub_weights  2-d array of sub-weights for vector elements divided
 *                      nonuniformly; each nonzero row must sum to 1. Elements
 *                      in rows corresponding to nonzero elements of n_div are
 *                      undefined. (dimensions \p count x \p n_sub)
 * \param [out] keep_idx 2-d array that contains 1's at all positions to be
 *                      preserved exactly. Elements must be zeroed before
 *                      calling. If vector is divided uniformly, only the
 *                      element in the 0th column is set to 1
 *                      (dimensions \p count * \p n_sub)
 * \param [in] sub_sizes    If non-null, sub_weights is treated as a jagged 2-D array, with \p sub_sizes
 *                        denoting the number of elements in each row
 * \param [in] count    Length of vector to compress
 * \param [in, out] n_samp Pointer to desired number of nonzero elements after
 *                      compression; upon return, points to remaining number
 *                      available for systematic resampling
 * \param [out] wt_remain Sum of magnitudes of sub-elements not preserved
 *                      exactly at each position (length \p count)
 * \return sum of magnitudes of elements on this local MPI process that are not
 * preserved exactly
 */
double find_keep_sub(double *values, unsigned int *n_div,
                     const Matrix<double> &sub_weights, Matrix<bool> &keep_idx,
                     uint16_t *sub_sizes,
                     size_t count, unsigned int *n_samp, double *wt_remain);


/*! \brief Perform systematic resampling on a vector with subdivided elements
 *
 * \param [in] values   Magnitudes of elements of original vector to be
 *                      compressed, potentially with some elements preserved
 *                      exactly (length \p count)
 * \param [in] n_div    Number of uniform intervals into which each element is
 *                      divided. If =0, this element is divided nonuniformly
 *                      according to sub_wts at this position (length \p count)
 * \param [in] sub_weights  2-d array of sub-weights for vector elements divided
 *                      nonuniformly; each nonzero row must sum to 1. Elements
 *                      in rows corresponding to nonzero elements of n_div are
 *                      undefined. (dimensions \p count * \p n_sub)
 * \param [in] keep_idx 2-d array that contains 1's at all positions to be
 *                      preserved exactly. (dimensions \p count * \p n_sub)
 * \param [in] sub_sizes    If non-null, sub_weights is treated as a jagged 2-D array, with \p sub_sizes
 *                        denoting the number of elements in each row
 * \param [in] count    Length of vector to compress
 * \param [in] n_samp   Number of elements to select in systematic resampling
 * \param [in] wt_remain Sum of magnitudes of sub-elements not preserved
 *                      exactly at each position (length \p count)
 * \param [in, out] loc_norms Sum of magnitudes of elements on each MPI process
 *                      not preserved exactly
 * \param [in] rand_num A random number chosen uniformly on [0, 1). Only the
 *                      argument from the 0th MPI process is used.
 * \param [out] new_vals Magnitudes of elements in compressed vector, including
 *                      elements preserved exactly
 * \param [out] new_idx Indices of elements of the compressed vector in the
 *                      original (input) vector. The 0th column gives the index
 *                      in the values array, and the 1st gives the index of the
 *                      subdivided element
 * \return Number of elements in compressed vector on this processor
 */
size_t sys_sub(double *values, unsigned int *n_div,
               const Matrix<double> &sub_weights, Matrix<bool> &keep_idx,
               uint16_t *sub_sizes,
               size_t count, unsigned int n_samp, double *wt_remain,
               double *loc_norms, double rand_num, double *new_vals,
               size_t new_idx[][2]);


/*! \brief Perform systematic compression with exact preservation on a vector
 * whose elements are divided into sub-weights
 *
 * \param [in] values   Magnitudes of elements of original vector on which to
 *                      perform compression (length \p count)
 * \param [in] count    Length of vector to compress
 * \param [in] n_div    Number of uniform intervals into which each element is
 *                      divided. If =0, this element is divided nonuniformly
 *                      according to \p sub_weights at this position (length \p count)
 * \param [in] sub_weights  2-d array of sub-weights for vector elements divided
 *                      nonuniformly; each nonzero row must sum to 1. Elements
 *                      in rows corresponding to nonzero elements of n_div are
 *                      undefined. (dimensions \p count * \p n_sub)
 * \param [in] keep_idx Scratch array used to identify elements to preserve
 *                      exactly. Must be 0 upon input
 *                      (dimensions \p count * \p n_sub)
 * \param [in] sub_sizes    If non-null, sub_weights is treated as a jagged 2-D array, with \p sub_sizes
 *                        denoting the number of elements in each row
 * \param [in] n_samp   Desired number of nonzero elements in compressed vector
 * \param [in] wt_remain Scratch array used for compression (length \p count)
 * \param [in] rand_num A random number chosen uniformly on [0, 1). Only the
 *                      argument from the 0th MPI process is used.
 * \param [out] new_vals Magnitudes of elements in compressed vector, including
 *                      elements preserved exactly
 * \param [out] new_idx Indices of elements of the compressed vector in the
 *                      original (input) vector. The 0th column gives the index
 *                      in the values array, and the 1st gives the index of the
 *                      subdivided element
 * \return number of elements in the compressed array (at most n_samp)
 */
size_t comp_sub(double *values, size_t count, unsigned int *n_div,
                Matrix<double> &sub_weights, Matrix<bool> &keep_idx,
                uint16_t *sub_sizes,
                unsigned int n_samp, double *wt_remain, double rand_num,
                double *new_vals, size_t new_idx[][2]);


/*! \brief Adjust energy shift to maintain one-norm of solution vector in DMC
 * simulation
 *
 * \param [in, out] shift Pointer to energy shift; updated upon return
 * \param [in] one_norm Current one-norm of solution vector
 * \param [in, out] last_norm Ptr to previous one-norm of solution vector, or 0
 *                      if vector norm is not yet being updated. Upon return,
 *                      set to \p one_norm if one_norm > target_norm
 * \param [in] target_norm One-norm above which the energy shift should be
 *                      adjusted
 * \param [in] damp_factor Factor by which the shift calculated based on the
 *                      change in one-norm is damped (or amplified)
 */
void adjust_shift(double *shift, double one_norm, double *last_norm,
                  double target_norm, double damp_factor);


/*! \brief Set-up for the alias method for multinomial sampling
 *
 * Calculates the alias for each state and the alternative probabilities
 * according to the algorithm in Figure 4 of Holmes et al. (2016)
 *
 * \param [in] probs        Normalized probabilities of choosing each state
 *                          (length \p n_states)
 * \param [out] aliases     Alias for each state (length \p n_states)
 * \param [out] alias_probs The probability of choosing the state i instead
 *                          of its alias aliases[i] (length \p n_states)
 * \param [in] n_states     The number of states
 */
void setup_alias(double *probs, unsigned int *aliases, double *alias_probs,
                 size_t n_states);


/*! \brief Perform multinomial sampling using the alias method
 *
 * \param [in] aliases      Alias for each state, calculated using setup_alias()
 *                          (length \p n_states)
 * \param [in] alias_probs  Alias probabilities for each state, calculated using
 *                          setup_alias(); need not be initialized
 *                          (length \p n_states)
 * \param [in] n_states     Number of states that can be sampled
 * \param [out] samples     The sampled indices, stored at intervals of
 *                          \p samp_int (length \p n_samp * samp_int)
 * \param [in] n_samp       Number of samples to draw multinomially
 * \param [in] samp_int     The interval at which to save samples in the
 *                          samples array
 * \param [in] mt_ptr   Address to MT state object to use for RN generation
 */
void sample_alias(unsigned int *aliases, double *alias_probs, size_t n_states,
                  uint8_t *samples, unsigned int n_samp, size_t samp_int,
                  mt_struct *mt_ptr);


#endif /* compress_utils_h */
