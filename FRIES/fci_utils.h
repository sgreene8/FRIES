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


/*! \brief Calculate the parity resulting from a double excitation
 *
 * The parity is (-1) raised to the power of the number of occupied orbitals
 * separating the occupied and virtual orbitals in the excitation
 *
 * \param [in] det  Pointer to origin determinant.
 * \param [in] orbs     The indices of the two occupied orbitals (0th
 *                      and 1st elements) and two virtual orbitals (2nd
 *                      and 3rd elements) defining the excitation
 *
 * \return parity of the excitation (+1 or -1)
 */
int doub_parity(uint8_t *det, uint8_t *orbs);


/*! \brief Calculate the determinant resulting from a double excitation
 *
 * \param [in, out] det Pointer to origin determinant. Upon return, contains new
 *                      determinant
 * \param [in] orbs     The indices of the two occupied orbitals (0th
 *                      and 1st elements) and two virtual orbitals (2nd
 *                      and 3rd elements) defining the excitation
 */
void doub_det(uint8_t *det, uint8_t *orbs);


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


 /*! \brief Calculate the parity resulting from a single excitation
  *
  * The parity is (-1) raised to the power of the number of occupied orbitals
  * separating the occupied and virtual orbitals in the excitation
  *
  * \param [in] det Pointer to origin determinant
  * \param [in] orbs     The indices of the occupied orbital (0th element) and
  *                      virtual orbital (1st element) defining the excitation
  *
  * \return parity of the excitation (+1 or -1)
  */
 int sing_parity(uint8_t *det, uint8_t *orbs);


 /*! \brief Calculate the determinant resulting from a single excitation
  *
  * \param [in, out] det Pointer to origin determinant. Upon return, contains new
  *                      determinant
  * \param [in] orbs     The indices of the occupied orbital (0th element) and
  *                      virtual orbital (1st element) defining the excitation
  */
 void sing_det(uint8_t *det, uint8_t *orbs);


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


/*! \brief Calculate the parity of a single excitation
 *
 * \param [in] occ_idx   The index (in \p occ_orbs ) of the orbital from which the electron is excited
 * \param [in] virt_orb     The virtual orbital to which the electron is excited
 * \param [in] occ_orbs     An ordered list of occupied orbitals in the determinant
 * \param [in] n_elec       The number of occupied orbitals in the determinant
 * \return parity of the excitation (+1 or -1)
 */
int excite_sign_occ(uint8_t occ_idx, uint8_t virt_orb, const uint8_t *occ_orbs, uint32_t n_elec);


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


/*! \brief Flip the spins of all electrons in a Slater determinant
 *
 * \param [in] det_in       The Slater determinant whose spins will be flipped
 * \param [out] det_out      The result of flipping spins
 * \param [in] n_orb        The number of spatial orbitals in the determinant
 */
void flip_spins(uint8_t *det_in, uint8_t *det_out, uint8_t n_orb);


/*! \brief Identify whether 2 Slater determinants are connected by a single or double excitation
 *
 * \param [in] str1     First Slater determinant
 * \param [in] str2     Second Slater determinant
 * \param [out] orbs        The orbitals defining the excitation. For a single excitation, orbs[0] is occupied in
 *  \p str1 and unoccupied in \p str2 ; orbs[1] is unoccupied in \p str1 and occupied in \p str2 .
 *  For a double excitation, orbs[0] and orbs[1] are occupied in \p str1 and unoccupied in \p str2 ; orbs[2]
 *  and orbs[3] are unoccupied in \p str1 and occupied in \p str2
 */
uint8_t find_excitation(const uint8_t *str1, const uint8_t *str2, uint8_t *orbs, uint8_t n_bytes);


/*! \brief Identify whether a Slater determinant is connected to its time-reversed partner by a double excitation
 *
 * \param [in] occ_orbs  List of occupied orbitals in the determinant
 * \param [in] n_orb     Number of spatial orbitals in the basis
 * \param [in] n_elec   Number of occupied orbitals in the determinant
 * \param [out] diff_idx    If the determinants are connected, contains the 2 indices in occ_orbs that differ
 * \return 1 if it is connected, 0 if not
 */
int tr_doub_connect(const uint8_t *occ_orbs, uint32_t n_orb, uint32_t n_elec, uint8_t *diff_idx);

    
#ifdef __cplusplus
}
#endif

#endif /* fci_utils_h */
