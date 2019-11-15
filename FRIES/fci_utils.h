/*! \file
 *
 * \brief Utilities for manipulating Slater determinants for FCI calculations
 */

#ifndef fci_utils_h
#define fci_utils_h

#include <stdio.h>
#include <stdlib.h>
#include <FRIES/det_store.h>


#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Generate the bit string representation of the Hartree-Fock
 * determinant
 *
 * THe HF determinant is defined as the determinant consisting of the N/2
 * lowest energy spin up and spin down orbitals
 *
 * \param [in] n_orb    The number of spatial orbitals in the basis
 * \param [in] n_elec   The number of electrons in the system (N)
 * \param [out] det     Upon return, contains bit-string representation of HF determinant
 */
void gen_hf_bitstring(unsigned int n_orb, unsigned int n_elec, uint8_t *det);


/*! \brief Calculate the parity and determinant resulting from a double
 * excitation
 *
 * The parity is (-1) raised to the power of the number of occupied orbitals
 * separating the occupied and virtual orbitals in the excitation
 *
 * \param [in, out] det Pointer to origin determinant. Upon return, contains new
 *                      determinant
 * \param [in] orbs     The indices of the two occupied orbitals (0th
 *                      and 1st elements) and two virtual orbitals (2nd
 *                      and 3rd elements) defining the excitation
 *
 * \return parity of the excitation (+1 or -1)
 */
int doub_det_parity(uint8_t *det, uint8_t *orbs);


/*! \brief Calculate the parity and determinant resulting from a single
 * excitation
 *
 * The parity is (-1) raised to the power of the number of occupied orbitals
 * separating the occupied and virtual orbitals in the excitation
 *
 * \param [in, out] det Pointer to origin determinant. Upon return, contains new
 *                      determinant
 * \param [in] orbs     The indices of the occupied orbital (0th element) and
 *                      virtual orbital (1st element) defining the excitation
 *
 * \return parity of the excitation (+1 or -1)
 */
int sing_det_parity(uint8_t *det, uint8_t *orbs);


/*! \brief Calculate the parity of a single excitation
 *
 * Determine the sign of cre_op^+ des_op |det>; adapted from the pyscf
 * subroutine pyscf.fci.cistring.cre_des_sign
 *
 * \param [in] cre_op   The orbital index of the creation operator acting on the
 *                      determinant
 * \param [in] des_op   The orbital index of the destruction operator acting on the
 *                      determinant
 * \param [in] det      Bit-string representation of the determinant being acted
 *                      upon
 * \return parity of the excitation (+1 or -1)
 */
int excite_sign(uint8_t cre_op, uint8_t des_op, uint8_t *det);

    
#ifdef __cplusplus
}
#endif

#endif /* fci_utils_h */
