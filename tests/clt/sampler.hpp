/*! \file
 * \brief Definition of an abstract class for repeatedly performing a sampling or compression operation
 */

#ifndef sampler_h
#define sampler_h

#include <vector>
#include <FRIES/ndarr.hpp>
#include <FRIES/compress_utils.hpp>

/*! \brief Abstract class definition for performing a generic compression or sampling operation
 */
class Sampler {
    uint32_t n_times_; ///< Total number of times the compression or sampling operation has been performed
public:
    /*! \brief Sample the target distribution or perform stochastic compression on the target vector
     */
    virtual void sample() {
        n_times_++;
    }
    
    /*! \brief Calculate the maximum difference between the cumulative mean
     * and the target distribution/vector
     */
    virtual double calc_max_diff() = 0;
    
    /*! \brief Constructor for abstract class
     */
    Sampler(size_t size, unsigned int n_samp) : accum_(size, 0), n_samp_(n_samp) {
        rngen_ptr_ = get_mt_parameter_id_st(32, 607, 0, (unsigned int) time(NULL));
        sgenrand_mt((uint32_t) time(NULL), rngen_ptr_);
    }
    
    ~Sampler() {
        free(rngen_ptr_);
    }
    
protected:
    mt_struct *rngen_ptr_; ///< Pointer to a struct for generating random numbers
    std::vector<double> accum_; ///< Vector in which to accumulate the cumulative mean
    unsigned int n_samp_; ///< Number of samples to use for compression
    
    /*! \brief Generate a random number from the uniform distribution on [0, 1)
     */
    double gen_rn() {
        return genrand_mt(rngen_ptr_) / (1. * UINT32_MAX);
    }
};


class HierComp : Sampler {
    std::vector<unsigned int> counts_; ///< Number of divisions for rows that are divided uniformly
    std::vector<double> row_wts_; ///< Sum of weights for each row
    Matrix<double> sub_wts_; ///< Probabilities for rows that are not divided uniformly
    Matrix<bool> keep_idx_;
    double loc_norm_; ///< Sum of magnitudes of elements not preserved exactly
    std::vector<double> wt_remain_; ///< Scratch space to use in compression
    std::vector<double> comp_vals_; ///< Vector to hold the compressed values from compression operation
    Matrix<size_t> comp_idx_; ///< 2-D array to hold indices of compressed values
public:
    HierComp(size_t rows, size_t cols, unsigned int n_samp) : counts_(rows), row_wts_(rows), sub_wts_(rows, cols), Sampler(rows * cols, n_samp), keep_idx_(rows, cols), wt_remain_(rows), comp_idx_(n_samp, 2) {
        for (size_t row_idx = 0; row_idx < rows; row_idx += 2) {
            row_wts_[row_idx] = Sampler::gen_rn() * 10;
            counts_[row_idx] = 0;
            double tot_weight = 0;
            for (size_t sub_idx = 0; sub_idx < cols; sub_idx++) {
                sub_wts_(row_idx, sub_idx) = Sampler::gen_rn();
                tot_weight += sub_wts_(row_idx, sub_idx);
            }
            for (size_t sub_idx = 0; sub_idx < cols; sub_idx++) {
                sub_wts_(row_idx, sub_idx) /= tot_weight;
            }
        }
        for (size_t row_idx = 1; row_idx < rows; row_idx += 2) {
            row_wts_[row_idx] = Sampler::gen_rn() * 10;
            counts_[row_idx] = genrand_mt(rngen_ptr_) % cols;
        }
        loc_norm_ = find_keep_sub(row_wts_.data(), counts_.data(), sub_wts_, keep_idx_, NULL, rows, &(Sampler::n_samp_), wt_remain_.data());
    }
    
    void sample() {
        Sampler::sample();
        double rn = Sampler::gen_rn();
        size_t n_cmp = sys_sub(row_wts_.data(), counts_.data(), sub_wts_, keep_idx_, NULL, row_wts_.size(), n_samp_, wt_remain_.data(), &loc_norm_, rn, comp_vals_.data(), (size_t (*)[2]) comp_idx_.data());
        for (size_t samp_idx = 0; samp_idx < n_cmp; samp_idx++) {
            accum_[comp_idx_(samp_idx, 0) * keep_idx_.cols() + comp_idx_(samp_idx, 1)] += comp_vals_[samp_idx];
        }
    }
    
    double calc_max_diff() {
        double max = 0;
        size_t n_col = sub_wts_.cols();
        for (size_t row_idx = 0; row_idx < row_wts_.size(); row_idx++) {
            if (counts_[row_idx] == 0) {
                for (size_t sub_idx = 0; sub_idx < n_col; sub_idx++) {
                    double diff = fabs(accum_[row_idx * n_col + sub_idx] - row_wts_[row_idx] * sub_wts_(row_idx, sub_idx));
                    if (diff > max) {
                        max = diff;
                    }
                }
            }
            else {
                for (size_t sub_idx = 0; sub_idx < counts_[row_idx]; sub_idx++) {
                    double diff = fabs(accum_[row_idx * n_col + sub_idx] - row_wts_[row_idx] / counts_[row_idx]);
                    if (diff > max) {
                        max = diff;
                    }
                }
            }
        }
        return max;
    }
};

#endif /* sampler_h */
