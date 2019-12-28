/*! \file
 *
 * \brief Utilities for Heat-Bath Power-Pitzer compression of Hamiltonian
 *
 * These functions apply only to the double excitation portion of a molecular
 * Hamiltonian matrix. Single excitations are treated as in the near-uniform
 * scheme
 */

#ifndef heat_bathPP_h
#define heat_bathPP_h

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <FRIES/compress_utils.hpp>
#include <FRIES/ndarr.hpp>


/*!
 * \brief Data structure used to store the elements defining the intermediate
 * matrices in the HB-PP factorization
 */
struct hb_info {
    size_t n_orb; ///< Number of spatial orbitals in the Hartree-Fock Basis
    double *s_tens; ///< Single electron components of HB-PP factorization, length n_orb
    double s_norm; ///< One-norm of single-electron components
    double *d_same; ///< Double-electron components of HB-PP factorization for same spins, stored as a 1-D array of length (n_orb choose 2)
    double *d_diff; ///< Double-electron components of HB-PP factorization for different spins, stored as a 2-D array of dimensions (n_orb x n_orb)
    double *exch_sqrt; ///< Square roots of exchange integrals <ia|ai>, stored as a 1-D array of length (n_orb choose 2)
    double *diag_sqrt; ///< Square roots of diagonal eris < p p | p p>, length n_orb
    double *exch_norms; ///< Row sums of the matrix of square roots of exchange integrals
};


/*! \brief Calculate terms in the HB-PP factorization from two-electron integrals from
 * Hartree-Fock
 *
 * Double-electron components are defined for \f$ p \neq q \f$ as
 * \f[
 * D_{pq} = \sum_{r,s \notin \lbrace p, q \rbrace} \left\lvert \langle p q || r s \rangle \right\rvert
 * \f]
 * Single-electron components are defined as \f$ S_p = \sum_{q} D_{pq} \f$
 *
 * \param [in] tot_orb      number of spatial orbitals from Hartree-Fock
 * \param [in] n_orb        number of unfrozen spatial orbitals to use in
 *                          constructing single- and double-electron components
 * \param [in] eris         2-electron integrals
 * \return hb_info struct initialized with the HB-PP parameters
 */
hb_info *set_up(unsigned int tot_orb, unsigned int n_orb,
                const FourDArr &eris);


/*! \brief Calculate the normalized probabilities for choosing the first
 * occupied orbital in a double excitation from a particular determinant
 *
 * The probabilities for the first occupied orbital are given as:
 * \f[
 *  P_i = \frac{S_i}{\sum_{j \in \text{occ}} S_j}
 * \f]
 *
 * \param [in] tens         hb_info struct containing the HB-PP parameters
 * \param [out] prob_arr    probability for each occupied orbital
 *                          (length \p n_elec)
 * \param [in] n_elec       Number of electrons in the determinant
 * \param [in] occ_orbs     array containing the indices of occupied orbitals in
 *                          the determinant (length \p n_elec)
 * \return sum of all single-electron weights before normalization
 */
double calc_o1_probs(hb_info *tens, double *prob_arr, unsigned int n_elec,
                     uint8_t *occ_orbs);


/*! \brief Calculate the normalized probabilities for choosing the second
 * occupied orbital in a double excitation from a particular determinant
 *
 * The probabilities for the second occupied orbital given the selection of the
 * first occupied orbital \f$ i \f$ are
 * \f[
 * P_j = \frac{D_{ij}}{\sum_{j' \in \text{occ}} D_{ij'}}
 * \f]
 *
 * \param [in] tens         hb_info struct containing the HB-PP parameters
 * \param [out] prob_arr    probability for each occupied orbital
 *                          (length \p n_elec)
 * \param [in] n_elec       Number of electrons in the determinant
 * \param [in] occ_orbs     array containing the indices of occupied orbitals in
 *                          the determinant (length \p n_elec)
 * \param [in,out] o1       Ponter to the index of the first occupied orbital.
 *                          Upon return, pointer to the orbital itself
 * \return sum of all single-electron weights before normalization
 */
double calc_o2_probs(hb_info *tens, double *prob_arr, unsigned int n_elec,
                     uint8_t *occ_orbs, uint8_t *o1);


/*! \brief Calculate the normalized probabilities for choosing the first
 * unoccupied orbital in a double excitation from a particular determinant
 *
 * The probabilities for the first unoccupied orbital are
 * \f[
 * P_a = \frac{|\langle i a | a i \rangle|^{1/2}}{\sum_{c \in \lbrace \text{virt} \rbrace} |\langle i c | c i \rangle|^{1/2}}
 * \f]
 * where i is the first occupied orbital selected previously.
 *
 * \param [in] tens         hb_info struct containing the HB-PP parameters
 * \param [out] prob_arr    probability for each unoccupied orbital (length n_orb)
 * \param [in] o1_orb       first occupied orbital (selected previously)
 * \param [in] det          bit-string representation of determinant from which
 *                          excitations occur
 * \return sum of weights for all unoccupied orbitals before normalization
 */
double calc_u1_probs(hb_info *tens, double *prob_arr, uint8_t o1_orb,
                     uint8_t *det);


/*! \brief Calculate the normalized probabilities for choosing the second
 * unoccupied orbital in a double excitation from a particular determinant
 *
 * The probabilities are defined similarly as for the first unoccupied orbital,
 * except that the sum is over the symmetry-allowed orbitals (not necessarily
 * unoccupied) in the basis
 *
 * \param [in] tens         hb_info struct containing the HB-PP parameters
 * \param [out] prob_arr    probability for each unoccupied orbital
 * \param [in] o1_orb       first occupied orbital (selected previously)
 * \param [in] o2_orb       second occupied orbital (selected previously)
 * \param [in] u1_orb       first unoccupied orbital (selected previously)
 * \param [in] lookup_tabl  List of orbitals with each type of symmetry, as
 *                          generated by gen_symm_lookup()
 * \param [in] symm         Irreps of the HF orbitals in the basis
 *                          (length n_orb)
 * \param [in, out] prob_len If points to 0, set to the number of irreps in the
 *                          corresponding row of \p lookup_tabl; Otherwise,
 *                          taken to indicate the length of \p prob_arr, such
 *                          additional elements in prob_len beyond the number of
 *                          irreps are zeroed.
 * \return sum of weights for all unoccupied orbitals before normalization
 */
double calc_u2_probs(hb_info *tens, double *prob_arr, uint8_t o1_orb,
                     uint8_t o2_orb, uint8_t u1_orb,
                     const Matrix<uint8_t> &lookup_tabl, uint8_t *symm,
                     unsigned int *prob_len);


/*! \brief Calculate the unnormalized weight for a double excitation from a
 *  particular determinant
 *
 * This function accounts for the different possible ways a double excitation
 * can be selected, based on the order of the occupied and unoccupied orbitals.
 *
 * \param [in] tens         hb_info struct containing the HB-PP parameters
 * \param [in] orbs         Array containing the 2 occupied and 2 unoccupied
 *                          orbitals defining the excitation
 * \return the weight for the excitation
 */
double calc_unnorm_wt(hb_info *tens, uint8_t *orbs);


/*! \brief Calculate the normalized weight for a double excitation from a
 *  particular determinant
 *
 * This function accounts for the different possible ways a double excitation
 * can be selected, based on the order of the occupied and unoccupied orbitals.
 * The weights are normalized at each level of the hierarchical construction.
 *
 * \param [in] tens         hb_info struct containing the HB-PP parameters
 * \param [in] orbs         Array containing the 2 occupied and 2 unoccupied
 *                          orbitals defining the excitation
 * \param [in] occ          List of occupied orbitals in \p det
 * \param [in] n_elec       Number of occupied orbitals in \p det
 * \param [in] det          Origin determinant for the excitation
 * \param [in] lookup_tabl   List of orbitals with each type of symmetry, as
 *                          generated by gen_symm_lookup()
 * \param [in] symm         Irreps of the HF orbitals in the basis
 *                          (length n_orb)
 * \return the weight for the excitation
 */
double calc_norm_wt(hb_info *tens, uint8_t *orbs, uint8_t *occ,
                    unsigned int n_elec, uint8_t *det,
                    const Matrix<uint8_t> &lookup_tabl, uint8_t *symm);


/*! \brief Perform multinomial sampling on the double excitation part of the
 *  Hamiltonian matrix using the heat-bath Power-Pitzer factorization
 *
 * \param [in] det          bit string representation of the origin determinant
 *                          defining the respective column of the Hamiltonian
 * \param [in] occ_orbs     list of occupied orbitals in det
 *                          (length \p num_elec)
 * \param [in] num_elec     number of occupied orbitals in det
 * \param [in] tens         hb_info struct containing the HB-PP parameters
 * \param [in] orb_symm     irreps of the HF orbitals in the basis
 *                          (length \p num_orb)
 * \param [in] lookup_tabl   List of orbitals with each type of symmetry, as
 *                          generated by gen_symm_lookup()
 * \param [in] num_sampl    number of double excitations to sample from the
 *                          column
 * \param [in] rn_ptr       Pointer to an mt_struct object for RN generation
 * \param [out] chosen_orbs Contains occupied (0th and 1st columns) and
 *                          unoccupied (2nd and 3rd columns) orbitals for each
 *                          excitation (dimensions num_sampl x 4)
 * \param [out] prob_vec    Contains probability for each excitation
 *                          (length num_sampl)
 * \return number of excitations sampled (less than num_sampl if some excitations
 *  were null)
 */
unsigned int hb_doub_multi(uint8_t *det, uint8_t *occ_orbs,
                           unsigned int num_elec, uint8_t *orb_symm,
                           hb_info *tens, const Matrix<uint8_t> &lookup_tabl,
                           unsigned int num_sampl, mt_struct *rn_ptr,
                           uint8_t (* chosen_orbs)[4], double *prob_vec);


#endif /* heat_bathPP_h */
