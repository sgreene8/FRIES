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
    uint32_t n_times_; ///< Total number of times the compression or sampling operation has been performed
    
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
    Matrix<bool> keep_idx1_;
    Matrix<bool> keep_idx2_;
    double loc_norm_; ///< Sum of magnitudes of elements not preserved exactly
    std::vector<double> wt_remain_; ///< Scratch space to use in compression
    std::vector<double> comp_vals_; ///< Vector to hold the compressed values from compression operation
    Matrix<size_t> comp_idx_; ///< 2-D array to hold indices of compressed values
public:
    HierComp(size_t rows, size_t cols, unsigned int n_samp) : counts_(rows), row_wts_(rows), sub_wts_(rows, cols), Sampler(rows * cols, n_samp), keep_idx1_(rows, cols), keep_idx2_(rows, cols), wt_remain_(rows), comp_idx_(n_samp, 2), comp_vals_(n_samp) {
        for (size_t row_idx = 0; row_idx < rows; row_idx += 2) {
            row_wts_[row_idx] = gen_rn() * 10;
            counts_[row_idx] = 0;
            double tot_weight = 0;
            for (size_t sub_idx = 0; sub_idx < cols; sub_idx++) {
                sub_wts_(row_idx, sub_idx) = gen_rn();
                tot_weight += sub_wts_(row_idx, sub_idx);
            }
            for (size_t sub_idx = 0; sub_idx < cols; sub_idx++) {
                sub_wts_(row_idx, sub_idx) /= tot_weight;
            }
        }
        for (size_t row_idx = 1; row_idx < rows; row_idx += 2) {
            row_wts_[row_idx] = gen_rn() * 10;
            counts_[row_idx] = (genrand_mt(rngen_ptr_) % cols) + 1;
        }
        loc_norm_ = find_keep_sub(row_wts_.data(), counts_.data(), sub_wts_, keep_idx1_, NULL, rows, &(Sampler::n_samp_), wt_remain_.data());
    }
    
    void sample() override {
        Sampler::sample();
        double rn = gen_rn();
        std::copy(keep_idx1_.data(), keep_idx1_.data() + keep_idx1_.cols() * row_wts_.size(), keep_idx2_.data());
        size_t n_cmp = sys_sub(row_wts_.data(), counts_.data(), sub_wts_, keep_idx2_, NULL, row_wts_.size(), n_samp_, wt_remain_.data(), &loc_norm_, rn, comp_vals_.data(), (size_t (*)[2]) comp_idx_.data());
        for (size_t samp_idx = 0; samp_idx < n_cmp; samp_idx++) {
            accum_[comp_idx_(samp_idx, 0) * keep_idx1_.cols() + comp_idx_(samp_idx, 1)] += comp_vals_[samp_idx];
        }
    }
    
    double calc_max_diff() override {
        double max = 0;
        size_t n_col = sub_wts_.cols();
        for (size_t row_idx = 0; row_idx < row_wts_.size(); row_idx++) {
            if (counts_[row_idx] == 0) {
                for (size_t sub_idx = 0; sub_idx < n_col; sub_idx++) {
                    double diff = fabs(accum_[row_idx * n_col + sub_idx] / n_times_ - row_wts_[row_idx] * sub_wts_(row_idx, sub_idx));
                    if (diff > max) {
                        max = diff;
                    }
                }
            }
            else {
                for (size_t sub_idx = 0; sub_idx < counts_[row_idx]; sub_idx++) {
                    double diff = fabs(accum_[row_idx * n_col + sub_idx] / n_times_ - row_wts_[row_idx] / counts_[row_idx]);
                    if (diff > max) {
                        max = diff;
                    }
                }
            }
        }
        return max;
    }
};


class NonuniComp : Sampler {
    std::vector<double> orig_vec_;
    std::vector<double> tmp_vec_;
    std::vector<size_t> srt_idx_;
    std::vector<bool> keep_idx1_;
    std::vector<bool> keep_idx2_;
    double loc_norm1_;
    double loc_norm2_;
    std::vector<double> probs_;
public:
    NonuniComp(size_t n_elem, unsigned int n_samp) : orig_vec_(n_elem), tmp_vec_(n_elem), srt_idx_(n_elem), keep_idx1_(n_elem), keep_idx2_(n_elem), Sampler(n_elem, n_samp), probs_(10) {
        for (size_t el_idx = 0; el_idx < n_elem; el_idx++) {
            orig_vec_[el_idx] = gen_rn() * 2 - 1;
            srt_idx_[el_idx] = el_idx;
        }
        double glob_norm;
        loc_norm1_ = find_preserve(orig_vec_.data(), srt_idx_.data(), keep_idx1_, n_elem, &n_samp_, &glob_norm);
        double prob_sum = 0;
        for (double &x : probs_) {
            x = gen_rn();
            prob_sum += x;
        }
        for (double &x : probs_) {
            x /= prob_sum;
        }
    }
    
    
    void sample() override {
        Sampler::sample();
        std::copy(keep_idx1_.begin(), keep_idx1_.end(), keep_idx2_.begin());
        std::copy(orig_vec_.begin(), orig_vec_.end(), tmp_vec_.begin());
        loc_norm2_ = loc_norm1_;
        double rn = gen_rn();
        sys_comp_nonuni(tmp_vec_.data(), tmp_vec_.size(), &loc_norm2_, n_samp_, keep_idx2_, probs_.data(), probs_.size(), rn);
        for (size_t el_idx = 0; el_idx < tmp_vec_.size(); el_idx++) {
            accum_[el_idx] += tmp_vec_[el_idx];
        }
    }
    
    
    double calc_max_diff() override {
        double max = 0;
        for (size_t el_idx = 0; el_idx < orig_vec_.size(); el_idx++) {
            double diff = fabs(accum_[el_idx] / n_times_ - orig_vec_[el_idx]);
            if (diff > max) {
                max = diff;
            }
        }
        return max;
    }
};

class ParBudget : Sampler {
    std::vector<double> norms_;
    double tot_norm_;
    double loc_norm_;
public:
    ParBudget(unsigned int n_samp) : Sampler(1, n_samp) {
        int n_procs = 1;
        int proc_rank = 0;
#ifdef USE_MPI
        MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
#endif
        norms_.reserve(n_procs);
        loc_norm_ = gen_rn();
        norms_[proc_rank] = loc_norm_;
#ifdef USE_MPI
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, norms_.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD);
#endif
        for (size_t el_idx = 0; el_idx < n_procs; el_idx++) {
            tot_norm_ += norms_[el_idx];
        }
    }
    
    void sample() override {
        Sampler::sample();
        uint32_t budget = sys_budget(norms_.data(), n_samp_, gen_rn());
        accum_[0] += budget;
    }
    
    double calc_max_diff() override {
        int n_procs = 1;
        int proc_rank = 0;
#ifdef USE_MPI
        MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
#endif
        double all_results[n_procs];
#ifdef USE_MPI
        MPI_Gather(accum_.data(), 1, MPI_DOUBLE, all_results, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
        double max = 0;
        if (proc_rank == 0) {
            for (int proc_idx = 0; proc_idx < n_procs; proc_idx++) {
                double diff = fabs(all_results[proc_idx] / n_times_ - n_samp_ * norms_[proc_idx] / tot_norm_);
                if (diff > max) {
                    max = diff;
                }
            }
        }
        return max;
    }
};

#endif /* sampler_h */
