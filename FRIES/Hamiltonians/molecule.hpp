/*! \file
 * \brief Utilities for a Hamiltonian describing a molecular system
 */

#ifndef molecule_h
#define molecule_h

#include <FRIES/fci_utils.h>
#include <FRIES/vec_utils.hpp>
#include <FRIES/ndarr.hpp>


/*! \brief number of irreps in the supported point groups */
#define n_irreps 8

/*! \brief Factorization (distribution) used to compress the Hamiltonian for a molecule
 * In the near-uniform distribution, the weights in each branch of the hierarchical matrix structure are uniform
 * In the heat-bath Power-Pitzer distribution, the weights are calculated based on the two-electron integrals
 * at the beginning of the calculation, and normalized for each determinant
 * In the unnormalized heat-bath Power-Pitzer distribution, the heat-bath weights are normalized for all determinants
 * at the beginning of the calculation
 */
typedef enum {
    near_uni,
    heat_bath,
    unnorm_heat_bath
} h_dist;

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
double doub_matr_el_nosgn(uint8_t *chosen_orbs, unsigned int n_orbs,
                          const FourDArr &eris, unsigned int n_frozen);

double doub_matr_el_nosgn(uint8_t *chosen_orbs, unsigned int n_orbs,
                          const SymmERIs &eris, unsigned int n_frozen);


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
double sing_matr_el_nosgn(uint8_t *chosen_orbs, uint8_t *occ_orbs,
                          unsigned int n_orbs, const FourDArr &eris,
                          const Matrix<double> &h_core, unsigned int n_frozen,
                          unsigned int n_elec);


double sing_matr_el_nosgn(uint8_t *chosen_orbs, uint8_t *occ_orbs,
                          unsigned int n_orbs, const SymmERIs &eris,
                          const Matrix<double> &h_core, unsigned int n_frozen,
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
size_t doub_ex_symm(uint8_t *det, uint8_t *occ_orbs, unsigned int num_elec,
                    unsigned int num_orb, uint8_t res_arr[][4], uint8_t *symm);


/*! \brief Generate all spin- and symmetry-allowed single excitations from a
 * Slater determinant.
 *
 * \param [in] det          Bit string representation of the origin determinant
 * \param [in] occ_orbs     Array of occupied orbitals in \p det
 *                          (length \p num_elec)
 * \param [in] num_elec     Number of occupied orbitals in the determinant
 * \param [in] num_orb      Number of unfrozen spatial HF orbitals in the basis
 * \param [out] res_arr     2-d array containing the occupied (0th) and
 *                          unoccupied (1st) orbitals defining each
 *                          excitation
 * \param [in] symm         irrep of each of the orbitals in the HF basis
 * \return total number of excitations generated
 */
size_t sing_ex_symm(uint8_t *det, uint8_t *occ_orbs, unsigned int num_elec,
                    unsigned int num_orb, uint8_t res_arr[][2], uint8_t *symm);

/*! \brief Deterministically multiply a vector by a 1-electron operator a^\dagger_p a_q
 *
 * \param [in, out]  vec    upon return, contains the vector obtained by multiplying by the 1-electron operator
 * \param [in] n_orbs       Number of HF unfrozen spatial orbitals in the basis
 * \param [in] des_op       Orbital index of the destruction operator component of the 1-electron operator
 * \param [in] cre_op       Orbital index of the creation operator component of the 1-electron operator
 * \param [in] dest_idx     The index of the sub-vector in \p vec to store the result in
 */
void one_elec_op(DistVec<double> &vec, unsigned int n_orbs, uint8_t des_op, uint8_t cre_op,
                 uint8_t dest_idx);


/*! \brief Deterministically multiply a vector by the off-diagonal elements of the Hamiltonian matrix times a constant factor
 * This method multiplies a sub-vector within the inputted DistVec object, as determined by its \p curr_vec_idx instance variable
 *
 * \param [in,out] vec      upon return, contains the vector obtained by multiplying by H
 * \param [in] symm         irrep of each of the orbitals in the HF basis
 * \param [in] n_orbs       Number of HF spatial orbitals (including frozen)
 *                          in the basis
 * \param [in] eris         4-D array of 2-electron integrals in spatial basis
 * \param [in] h_core       2-D array of 1-electron integrals in spatial basis
 * \param [in] orbs_scratch     Scratch array for storing the orbitals involved in the excitation
 *                      (row dimension must be at least max number of excitations from one determinant)
 * \param [in] n_frozen     Number of core electrons frozen in the calculation
 * \param [in] n_elec       Number of unfrozen electrons in the system
 * \param [in] dest_idx     The index of the sub-vector in \p vec to store the result in
 * \param [in] h_fac        The constant factor used in the multiplication
 * \param [in] spin_parity      1 if targeting even-spin states with time-reveral symmetry, -1 for odd-spin states, 0 if time-reversal symmetry is not used
 */
void h_op_offdiag(DistVec<double> &vec, uint8_t *symm, unsigned int n_orbs,
                  const FourDArr &eris, const Matrix<double> &h_core,
                  uint8_t *orbs_scratch, unsigned int n_frozen,
                  unsigned int n_elec, uint8_t dest_idx, double h_fac, int spin_parity);
void h_op_offdiag(DistVec<double> &vec, uint8_t *symm, unsigned int n_orbs,
                  const FourDArr &eris, const Matrix<double> &h_core,
                  uint8_t *orbs_scratch, unsigned int n_frozen,
                  unsigned int n_elec, uint8_t dest_idx, double h_fac);
void h_op_offdiag(DistVec<double> &vec, uint8_t *symm, unsigned int n_orbs,
                  const SymmERIs &eris, const Matrix<double> &h_core,
                  uint8_t *orbs_scratch, unsigned int n_frozen,
                  unsigned int n_elec, uint8_t dest_idx, double h_fac, int spin_parity);
void h_op_offdiag(DistVec<double> &vec, size_t vec_size, uint8_t *symm, unsigned int n_orbs,
                  const SymmERIs &eris, const Matrix<double> &h_core,
                  uint8_t *orbs_scratch, unsigned int n_frozen,
                  unsigned int n_elec, uint8_t dest_idx, double h_fac, int spin_parity);


/*! \brief Deterministically multiply a vector by the diagonal portion of (a * I + b * H), where
 * H is the Hamiltonian and I is the identity
 * This method multiplies a sub-vector within the inputted DistVec object, as determined by its \p curr_vec_idx instance variable
 *
 * \param [in,out] vec      upon return, contains the vector obtained by multiplying by H
 * \param [in] dest_idx     The index of the sub-vector in \p vec to store the result in
 * \param [in] id_fac       The constant pre-factor for the identity matrix
 * \param [in] h_fac        The constant pre-factor for the Hamiltonian matrix
 */
void h_op_diag(DistVec<double> &vec, uint8_t dest_idx, double id_fac, double h_fac);


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
size_t gen_hf_ex(uint8_t *hf_det, uint8_t *hf_occ, unsigned int num_elec,
                 unsigned int n_orb, uint8_t *orb_symm, const FourDArr &eris,
                 unsigned int n_frozen, Matrix<uint8_t> &ex_dets, double *ex_mel);


/*! \brief Calculate number of double excitations from any determinant
 * accounting only for spin symmetry
 *
 * \param [in] num_elec     Number of electrons in the system
 * \param [in] num_orb      Number of unfrozen spatial orbitals in the basis
 * \return number of double excitations
 */
size_t count_doub_nosymm(unsigned int num_elec, unsigned int num_orb);


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
double diag_matrel(const uint8_t *occ_orbs, unsigned int n_orbs,
                   const FourDArr &eris, const Matrix<double> &h_core,
                   unsigned int n_frozen, unsigned int n_elec);
double diag_matrel(const uint8_t *occ_orbs, unsigned int n_orbs,
                   const SymmERIs &eris, const Matrix<double> &h_core,
                   unsigned int n_frozen, unsigned int n_elec);


/*! \brief Find the nth virtual orbital in a determinant with a particular spin and point-group symmetry
 *
 * \param [in] det          Bit-string representation of the origin determinant
 * \param [in] spin_orbs     Spin of the orbital in question (0 or 1) times the number of spatial orbitals in the basis
 * \param [in] irrep        The irrep of the orbital in question
 * \param [in] n        The index of the virtual orbital in question
 * \param [in] lookup_tabl  List of orbitals with each type of symmetry, as
 *                          generated by gen_symm_lookup()
 * \returns The orbital index of the virtual orbital in question
 */
uint8_t find_nth_virt_symm(uint8_t *det, uint8_t spin_orbs, uint8_t irrep, uint8_t n,
                           const Matrix<uint8_t> &lookup_tabl);


/*! \brief Generate a list of all spatial orbitals of each irrep in the point
 * group
 *
 * \param [in] orb_symm     Irrep of each orbital in the HF basis
 *                          (length \p n_orbs)
 * \param [out] lookup_tabl matrix whose 0th column lists number of orbitals
 *                          with each irrep, and subsequent columns list those
 *                          orbitals (dimensions n_symm x (\p n_orb + 1))
 */
void gen_symm_lookup(uint8_t *orb_symm, Matrix<uint8_t> &lookup_tabl);


/*! \brief Contains information about the symmetries of orbitals in a single-particle basis
 */
struct SymmInfo {
    std::vector<uint8_t> symm_vec; ///< Symmetry labels for each of the orbitals in the basis
    Matrix<uint8_t> symm_lookup; ///< Lookup table generated by gen_symm_lookup
    uint32_t max_n_symm; ///< The maximum number of orbitals within a symmetry group
    
    SymmInfo(uint8_t *symm, uint32_t n_orb) : symm_vec(n_orb), symm_lookup(n_irreps, n_orb + 1),
    max_n_symm(0) {
        std::copy(symm, symm + n_orb, symm_vec.begin());
        gen_symm_lookup(symm, symm_lookup);
        for (uint8_t symm_idx = 0; symm_idx < n_irreps; symm_idx++) {
            if (symm_lookup[symm_idx][0] > max_n_symm) {
                max_n_symm = symm_lookup[symm_idx][0];
            }
        }
    }
};


/*! \brief Count the number of spin- and symmetry-allowed single excitations
 * from a determinant.
 *
 * \param [in] det          Bit-string representation of the origin determinant
 * \param [in] occ_orbs     Array of occupied orbitals in \p det
 *                          (length \p num_elec)
 * \param [in] num_elec     Number of occupied orbitals in the determinant
 * \param [in] symm     Object containing information about the symmetries of orbitals in the single-particle basis
 * \return number of symmetry-allowed single excitations
 */
size_t count_singex(uint8_t *det, const uint8_t *occ_orbs, uint32_t num_elec,
                    SymmInfo *symm);


/*! \brief Print the orbitals in the spatial basis with each irrep
 *
 * \param [in] lookup_tabl  Symmetry table generated by gen_symm_lookup()
 */
void print_symm_lookup(Matrix<uint8_t> &lookup_tabl);

#endif /* molecule_h */
