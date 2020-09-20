/*! \file
* \brief Wrappers for handling calls to functions in the LAPACK llibraries
*/

#include "lapack_wrappers.hpp"
#if __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include "LAPACK/lapacke.h"
#endif

void get_svals(Matrix<double> mat, std::vector<double> &s_vals, double *scratch) {
    int rows = (int) mat.rows();
    int cols = (int) mat.cols();
    int e_size = MIN(rows, cols) - 1;
    double *e_scratch = scratch;
    double *tauq_scratch = e_scratch + e_size;
    double *taup_scratch = tauq_scratch + MIN(rows, cols);
    double *work_scratch = taup_scratch + MIN(rows, cols);
    int work_size = 32 * (rows + cols);
    int output;
    dgebrd_(&rows, &cols, mat.data(), &rows, s_vals.data(), e_scratch, tauq_scratch, taup_scratch, work_scratch, &work_size, &output);
    if (output != 0) {
        std::stringstream msg;
        msg << "Error in bidiagonal reduction step of SVD, return value is " << output;
        throw std::runtime_error(msg.str());
    }
    
    int n_sing_vecs = 0;
    int vt_leaddim = 1;
    char ul = 'U';
    dbdsqr_(&ul, &rows, &n_sing_vecs, &n_sing_vecs, &n_sing_vecs, s_vals.data(), e_scratch, nullptr, &vt_leaddim, nullptr, &vt_leaddim, nullptr, &vt_leaddim, mat.data(), &output);
    if (output != 0) {
        std::stringstream msg;
        msg << "Error in second step of SVD, return value is " << output;
        throw std::runtime_error(msg.str());
    }
}

void get_real_gevals_vecs(Matrix<double> op, Matrix<double> ovlp, std::vector<double> &real_evals,
                          Matrix<double> &real_evecs, double *scratch) {
    char compute_left = 'V';
    char compute_right = 'N';
    int rows = (int) op.rows();
    double *alpha_r = scratch;
    double *alpha_i = alpha_r + rows;
    double *beta = alpha_i + rows;
    int output;
    double *work_scratch = beta + rows;
    int vl_leaddim = 1;
    int work_size = 39 * rows;
    dggev_(&compute_left, &compute_right, &rows, op.data(), &rows, ovlp.data(), &rows, alpha_r, alpha_i, beta, real_evecs.data(), &rows, nullptr, &vl_leaddim, work_scratch, &work_size, &output);
    if (output != 0) {
        std::stringstream msg;
        msg << "Error in solving generalized eigenvalue problem, return value is " << output;
        throw std::runtime_error(msg.str());
    }
    
    for (uint8_t idx = 0; idx < rows; idx++) {
        real_evals[idx] = alpha_r[idx] / beta[idx];
    }
}
