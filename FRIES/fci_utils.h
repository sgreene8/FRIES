/*! \file
 *
 * \brief Utilities for manipulating Slater determinants for FCI calculations
 */

#ifndef fci_utils_h
#define fci_utils_h

#include <stdlib.h>
#include <stdint.h>

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


/*! \brief Given an ordered list of occupied orbitals and the orbitals involved in a single excitation, generate
 * a new ordered list of orbitals after excitation
 *
 * \param [in] curr_orbs    List of occupied orbitals in starting determinant
 * \param [out] new_orbs     List of occupied orbitals after excitation
 * \param [in] ex_orbs      An array of 4 numbers, the first two of which are the indices of the occupied
 * orbitals in the excitation, and the last two of which are the virtual orbitals themselves
 * \param [in] n_elec       The number of occupied orbitals
 */
void doub_ex_orbs(uint8_t *curr_orbs, uint8_t *new_orbs, uint8_t *ex_orbs,
                  uint8_t n_elec);


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


/*! \brief Given an ordered list of occupied orbitals and the orbitals involved in a single excitation, generate
 * a new ordered list of orbitals after excitation
 *
 * \param [in] curr_orbs    List of occupied orbitals in starting determinant
 * \param [out] new_orbs     List of occupied orbitals after excitation
 * \param [in] ex_orbs      An array of 2 numbers, the first of which is the index of the occupied orbital in
 * the excitation, and the second of which is the virtual orbital itself
 * \param [in] n_elec       The number of occupied orbitals
 */
void sing_ex_orbs(uint8_t *curr_orbs, uint8_t *new_orbs, uint8_t *ex_orbs,
                  uint8_t n_elec);


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


/*! \brief Find the orbital index of the nth virtual orbital with a given spin in a determinant
 *
 * \param [in] occ_orbs     List of the occupied orbitals in \p det
 * \param [in] spin     Spin of the orbital in question (0 or 1)
 * \param [in] n_elec       Number of electrons in the determinant
 * \param [in] n_orb        Number of spatial orbitals the basis
 * \param [in] n        The index of the virtual orbital in question
 * \returns The orbital index of the virtual orbital in question
 */
uint8_t find_nth_virt(uint8_t *occ_orbs, int spin, uint8_t n_elec,
                      uint8_t n_orb, uint8_t n);

    
#ifdef __cplusplus
}
#endif

#endif /* fci_utils_h */
