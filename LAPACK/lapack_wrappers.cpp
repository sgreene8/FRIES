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
                          Matrix<double> &real_evecs) {
    char compute_left = 'V';
    char compute_right = 'N';
    int rows = (int) op.rows();
    double alpha_r[rows];
    double alpha_i[rows];
    double beta[rows];
    int output;
    double work_scratch[39 * rows];
    int vl_leaddim = 1;
    int work_size = 39 * rows;
    double ray_quo[rows * rows];
    std::copy(op.data(), op.data() + rows * rows, ray_quo);
    double overlap[rows * rows];
    std::copy(ovlp.data(), ovlp.data() + rows * rows, overlap);
    double evecs[rows * rows];
    dggev_(&compute_left, &compute_right, &rows, ray_quo, &rows, overlap, &rows, alpha_r, alpha_i, beta, evecs, &rows, nullptr, &vl_leaddim, work_scratch, &work_size, &output);
    if (output != 0) {
        std::stringstream msg;
        msg << "Error in solving generalized eigenvalue problem, return value is " << output;
        throw std::runtime_error(msg.str());
    }
    
    for (uint8_t idx = 0; idx < rows; idx++) {
        real_evals[idx] = alpha_r[idx] / beta[idx];
    }
    std::copy(evecs, evecs + rows * rows, real_evecs.data());
}


void inv_inplace(Matrix<double> &mat, double *scratch) {
    int rows = (int) mat.rows();
    int pivots[mat.rows()];
    int output;
    dgetrf_(&rows, &rows, mat.data(), &rows, pivots, &output);
    if (output != 0) {
        std::stringstream msg;
        msg << "Error in computing LU factorization, return value is " << output;
        throw std::runtime_error(msg.str());
    }
    int work_size = rows * rows;
    dgetri_(&rows, mat.data(), &rows, pivots, scratch, &work_size, &output);
    if (output != 0) {
        std::stringstream msg;
        msg << "Error in computing inverse, return value is " << output;
        throw std::runtime_error(msg.str());
    }
}

void invu_inplace(Matrix<double> &mat, double *scratch) {
    int rows = (int) mat.rows();
    double lmat[rows * rows];
    double umat[rows * rows];
    std::fill(lmat, lmat + rows * rows, 0);
    std::copy(mat.data(), mat.data() + rows * rows, umat);
    for (size_t k = 0; k < rows; k++) {
        lmat[k * rows + k] = 1;
    }
    for (size_t k = 0; k < rows; k++) {
        for (size_t j = k + 1; j < rows; j++) {
            lmat[j * rows + k] = umat[j * rows + k] / umat[k * rows + k];
            if (fabs(umat[k * rows + k]) < 1e-5) {
                std::cerr << "Division by small number (" << umat[k * rows + k] << ") in inv(U) method\n";
            }
        }
        
        for (size_t l = k + 1; l < rows; l++) {
            for (size_t j = 0; j < rows; j++) {
                umat[l * rows + j] -= lmat[l * rows + k] * umat[k * rows + j];
            }
        }
    }
    
    double lapack_mat[rows * rows];
    int pivots[rows];
    std::fill(lapack_mat, lapack_mat + rows * rows, 0);
    for (size_t i = 0; i < rows; i++) {
        pivots[i] = (int) i + 1;
        for (size_t j = i; j < rows; j++) {
            lapack_mat[j * rows + i] = umat[i * rows + j];
        }
    }
    
    int work_size = rows * rows;
    int output;
    dgetri_(&rows, lapack_mat, &rows, pivots, scratch, &work_size, &output);
    if (output != 0) {
        std::stringstream msg;
        msg << "Error in computing inverse of U, return value is " << output;
        throw std::runtime_error(msg.str());
    }
    
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < rows; j++) {
            mat(i, j) = lapack_mat[j * rows + i];
        }
    }
}

void invr_inplace(Matrix<double> &mat, double *scratch) {
    int rows = (int) mat.rows();
    double *tau = scratch;
    double *work = tau + rows;
    int workdim = rows * rows;
    double *lapack_mat = work + workdim;
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < rows; j++) {
            lapack_mat[i * rows + j] = mat(j, i);
        }
    }
    int output;
    dgeqrf_(&rows, &rows, lapack_mat, &rows, tau, work, &workdim, &output);
    if (output != 0) {
        std::stringstream msg;
        msg << "Error in computing QR decomposition, return value is " << output;
        throw std::runtime_error(msg.str());
    }
    int pivots[rows];
    for (size_t i = 0; i < rows; i++) {
        pivots[i] = (int) i + 1;
        for (size_t j = i + 1; j < rows; j++) {
            lapack_mat[i * rows + j] = 0;
        }
    }
    
    dgetri_(&rows, lapack_mat, &rows, pivots, work, &workdim, &output);
    if (output != 0) {
        std::stringstream msg;
        msg << "Error in computing inverse of R, return value is " << output;
        throw std::runtime_error(msg.str());
    }
    
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < rows; j++) {
            mat(i, j) = lapack_mat[j * rows + i];
        }
    }
}


void gen_qr(Matrix<double> &orth_mat, Matrix<double> &rmat, double *scratch) {
    int rows = (int) orth_mat.rows();
    int cols = (int) orth_mat.rows();
    int min_mn = MIN(rows, cols);
    double *tau = scratch;
    double *work = tau + min_mn;
    int workdim = rows * cols;
    int output;
    dgeqrf_(&rows, &cols, orth_mat.data(), &rows, tau, work, &workdim, &output);
    if (output != 0) {
        std::stringstream msg;
        msg << "Error in computing QR decomposition, return value is " << output;
        throw std::runtime_error(msg.str());
    }
    
    std::copy(orth_mat.data(), orth_mat.data() + rows * cols, rmat.data());
    for (uint8_t i = 0; i < rows; i++) {
        for (uint8_t j = i + 1; j < cols; j++) {
            rmat(i, j) = 0;
        }
    }
    
    dorgqr_(&rows, &cols, &min_mn, orth_mat.data(), &rows, tau, work, &workdim, &output);
    if (output != 0) {
        std::stringstream msg;
        msg << "Error in forming Q matrix, return value is " << output;
        throw std::runtime_error(msg.str());
    }
}
