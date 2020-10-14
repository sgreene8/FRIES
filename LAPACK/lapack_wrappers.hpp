/*! \file
 * \brief Wrappers for handling calls to functions in the LAPACK library
 */

#ifndef lapack_wrappers_hpp
#define lapack_wrappers_hpp

#include <FRIES/ndarr.hpp>
#include <FRIES/math_utils.h>
#include <iostream>
#include <stdexcept>

#ifndef MIN
#define MIN(a, b) (((a)<(b))?(a):(b))
#endif

/*! \brief Calculate the singular values of a matrix
 *
 * \param [in] mat      The matrix for which singular values will be calculated
 * \param [out] s_vals      The singular  values of the matrix
 * \param [in]  scratch     A block of memory to use in the calculation, must have length at least
 *                      3 * min(rows, columns) + rows * columns - 1 + 32 * (rows + columns)
 */
void get_svals(Matrix<double> mat, std::vector<double> &s_vals, double *scratch);


/*! \brief Calculate the real parts of the nonsymmetric generalized eigenvalues and eigenvectors of a pair of matrices
 *
 * \param [in] op       The left matrix in the generalized eigenvalue problem, representing e.g. Rayleigh quotients of an operator
 * \param [in] ovlp     The right matrix in the generalized eigenvalue problem, representing e.g. overlap matrix elements for a nonorthonormal basis
 * \param [out] real_evals      The real parts of the eigenvalues
 * \param [out] real_evecs      The real parts of the eigenvectors
 * \param [in] scratch       A block of memory to use in the calculation, must have length at least (42 * rows)
 */
void get_real_gevals_vecs(Matrix<double> op, Matrix<double> ovlp, std::vector<double> &real_evals,
                          Matrix<double> &real_evecs, double *scratch);

/*! \brief Invert a square matrix in-place
 *
 * \param [in] mat      The matrix to be inverted
 * \param [in] scratch      A block of memory to be used as scratch in the calculation, must have length at least (rows^2)
 */
void inv_inplace(Matrix<double> &mat, double *scratch);

void gen_qr(Matrix<double> &orth_mat, Matrix<double> &rmat, double *scratch);

#endif /* lapack_wrappers_hpp */
