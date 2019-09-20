/*! \file
 * \brief Utilities for a Hamiltonian describing a molecular system
 */

#ifndef molecule_h
#define molecule_h

#include <stdio.h>
#include <FRIES/fci_utils.h>


#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Calculate a double excitation matrix element
 *
 * The parity of the excitation, i.e. from the ordering of the orbitals, is not
 * accounted for in this function
 *
 * \param [in] chosen_orbs  The indices of the two occupied orbitals (0th
 *                          and 1st elements) and two virtual orbitals (2nd
 *                          and 3rd elements) defining the excitation
 * \param [in] n_orbs       Number of HF spatial orbitals (including frozen)
 *                          in the basis
 * \param [in] eris         4-D array of 2-electron integrals in spatial basis
 * \param [in] n_frozen     Number of core electrons frozen in the calculation
 * \return calculated matrix element
 */
double doub_matr_el_nosgn(unsigned char *chosen_orbs, unsigned int n_orbs,
                          double (* eris)[n_orbs][n_orbs][n_orbs], unsigned int n_frozen);


/*! \brief Calculate a single excitation matrix element
 *
 * The parity of the excitation, i.e. from the ordering of the orbitals, is not
 * accounted for in this function
 *
 * \param [in] chosen_orbs  The indices of the occupied orbital (0th element)
 *                          and virtual orbital (1st element) defining the
 *                          excitation
 * \param [in] occ_orbs     Orbitals occupied in the origin determinant
 *                          (length \p n_elec)
 * \param [in] n_orbs       Number of HF spatial orbitals (including frozen)
 *                          in the basis
 * \param [in] eris         4-D array of 2-electron integrals in spatial basis
 * \param [in] h_core       2-D array of 1-electron integrals in spatial basis
 * \param [in] n_frozen     Number of core electrons frozen in the calculation
 * \param [in] n_elec       Number of unfrozen electrons in the system
 * \return calculated matrix element
 */
double sing_matr_el_nosgn(unsigned char *chosen_orbs, unsigned char *occ_orbs,
                          unsigned int n_orbs, double (* eris)[n_orbs][n_orbs][n_orbs],
                          double (* h_core)[n_orbs], unsigned int n_frozen,
                          unsigned int n_elec);


/*! \brief Generate all spin- and symmetry-allowed double excitations from a
 * Slater determinant.
 *
 * \param [in] det          Bit string representation of the origin determinant
 * \param [in] occ_orbs     Array of occupied orbitals in \p det
 *                          (length \p num_elec)
 * \param [in] num_elec     Number of occupied orbitals in the determinant
 * \param [in] num_orb      Number of unfrozen spatial HF orbitals in the basis
 * \param [out] res_arr     2-d array containing the occupied (0th and 1st) and
 *                          unoccupied (2nd and 3rd) orbitals defining each
 *                          excitation
 * \param [in] symm         irrep of each of the orbitals in the HF basis
 * \return total number of excitations generated
 */
size_t doub_ex_symm(long long det, unsigned char *occ_orbs, unsigned int num_elec,
                    unsigned int num_orb, unsigned char res_arr[][4], unsigned char *symm);


/*! \brief Calculate the HF column of the FCI Hamiltonian
 *
 * \param [in] hf_det       Bit-string representation of the HF determinant
 * \param [in] hf_occ       Orbitals occupied in the HF determinant
 *                          (length \p num_elec)
 * \param [in] num_elec     Number of occupied orbitals in the determinant
 * \param [in] n_orb        Number of spatial orbitals in the basis (including
 *                          frozen)
 * \param [in] orb_symm     Irrep of each spatial orbital in the basis
 * \param [in] eris         4-D array of 2-electron integrals in spatial basis
 * \param [in] n_frozen     Number of core electrons frozen in the calculation
 * \param [in] ex_dets      Slater determinant indices of elements in the HF
 *                          column
 * \param [in] ex_mel       Elements in the column
 * \return total number of nonzero elements
 */
size_t gen_hf_ex(long long hf_det, unsigned char *hf_occ, unsigned int num_elec,
                 unsigned int n_orb, unsigned char *orb_symm, double (*eris)[n_orb][n_orb][n_orb],
                 unsigned int n_frozen, long long *ex_dets, double *ex_mel);


/*! \brief Calculate number of double excitations from any determinant
 * accounting only for spin symmetry
 *
 * \param [in] num_elec     Number of electrons in the system
 * \param [in] num_orb      Number of unfrozen spatial orbitals in the basis
 * \return number of double excitations
 */
size_t count_doub_nosymm(unsigned int num_elec, unsigned int num_orb);


/*! \brief Count the number of spin- and symmetry-allowed single excitations
 * from a determinant.
 *
 * \param [in] det          Bit-string representation of the origin determinant
 * \param [in] occ_orbs     Array of occupied orbitals in \p det
 *                          (length \p num_elec)
 * \param [in] orb_symm     Irrep of each of the orbitals in the HF basis
 * \param [in] num_orb      Number of unfrozen spatial HF orbitals in the basis
 * \param [in] lookup_tabl  List of orbitals with each type of symmetry, as
 *                          generated by gen_symm_lookup()
 * \param [in] num_elec     Number of occupied orbitals in the determinant
 * \return number of symmetry-allowed single excitations
 */
size_t count_singex(long long det, unsigned char *occ_orbs, unsigned char *orb_symm,
                    unsigned int num_orb, unsigned char (* lookup_tabl)[num_orb + 1],
                    unsigned int num_elec);


/*! \brief Calculate diagonal Hamiltonian matrix element for a Slater
 * determinant
 *
 * \param [in] occ_orbs     Array of occupied orbitals in the determinant
 *                          (length \p n_elec - \p n_frozen)
 * \param [in] n_orbs       Number of HF spatial orbitals (including frozen) in
 *                          the basis
 * \param [in] eris         4-D array of 2-electron integrals in spatial MO
 *                          basis
 * \param [in] h_core       2-D array of 1-electron integrals in spatial MO
 *                          basis
 * \param [in] n_frozen     number of core electrons frozen in the calculation
 * \param [in] n_elec       number of unfrozen electrons in the system
 * \return calculated matrix element
 */
double diag_matrel(unsigned char *occ_orbs, unsigned int n_orbs,
                   double (* eris)[n_orbs][n_orbs][n_orbs], double (* h_core)[n_orbs],
                   unsigned int n_frozen, unsigned int n_elec);


#ifdef __cplusplus
}
#endif

#endif /* molecule_h */
