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
#include <FRIES/compress_utils.hpp>
#include <FRIES/ndarr.hpp>
#include <FRIES/fci_utils.h>
#include <FRIES/Hamiltonians/near_uniform.hpp>
#include <FRIES/Hamiltonians/molecule.hpp>

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
hb_info *set_up(uint32_t tot_orb, uint32_t n_orb, const SymmERIs &eris);


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
 * \param [in] exclude_first    If 1, the first occupied orbital in the determinant is excluded
 * \return sum of all single-electron weights before normalization
 */
double calc_o1_probs(hb_info *tens, double *prob_arr, unsigned int n_elec,
                     uint8_t *occ_orbs, int exclude_first);


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
 * \param [in,out] o1_idx       The index of the first occupied orbital
 * \return sum of all single-electron weights before normalization
 */
double calc_o2_probs(hb_info *tens, double *prob_arr, unsigned int n_elec,
                     uint8_t *occ_orbs, uint8_t o1_idx);


/*! \brief Calculate the normalized probabilities for choosing the second occupied orbital in a double
 * excitation, with the constraint that the orbital is less than the first occupied orbital
 *
 * \param [in] tens         hb_info struct containing the HB-PP parameters
 * \param [out] prob_arr    probability for each occupied orbital
 *                          (length \p n_elec)
 * \param [in] n_elec       Number of electrons in the determinant
 * \param [in] occ_orbs     array containing the indices of occupied orbitals in
 *                          the determinant (length \p n_elec)
 * \param [in,out] o1_idx       The index of the first occupied orbital
 * \return sum of all single-electron weights before normalization
 */
 double calc_o2_probs_half(hb_info *tens, double *prob_arr, unsigned int n_elec,
                           uint8_t *occ_orbs, uint8_t o1_idx);


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
 * \param [in] occ_orbs          list of all occupied orbitals in the determinant
 * \param [in] n_elec       Number of electrons in the determinant
 * \param [in] exclude_first    If 1, the first occupied orbital in the determinant is excluded
 * \return sum of weights for all unoccupied orbitals before normalization
 */
double calc_u1_probs(hb_info *tens, double *prob_arr, uint8_t o1_orb,
                     uint8_t *occ_orbs, uint8_t n_elec, int exclude_first);


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
 * \param [in] symm     Object containing information about the symmetries of orbitals in the single-particle basis 
 * \param [in, out] prob_len If points to 0, set to the number of irreps in the
 *                          corresponding row of \p lookup_tabl; Otherwise,
 *                          taken to indicate the length of \p prob_arr, such
 *                          additional elements in prob_len beyond the number of
 *                          irreps are zeroed.
 * \return sum of weights for all unoccupied orbitals before normalization
 */
double calc_u2_probs(hb_info *tens, double *prob_arr, uint8_t o1_orb,
                     uint8_t o2_orb, uint8_t u1_orb,
                     SymmInfo *symm, uint16_t *prob_len);


/*! \brief Calculate normalized weights for the second unoccupied orbital, excluding occupied orbitals
 *  and requiring that the second unoccupied orbital index be less than that of the first
 *
 * \param [in] tens         hb_info struct containing the HB-PP parameters
 * \param [out] prob_arr    probability for each unoccupied orbital
 * \param [in] o1_orb       first occupied orbital (selected previously)
 * \param [in] o2_orb       second occupied orbital (selected previously)
 * \param [in] u1_orb       first unoccupied orbital (selected previously)
 * \param [in] det          Bit-string representation of determinant
 * \param [in] symm     Object containing information about the symmetries of orbitals in the single-particle basis
 * \param [in, out] prob_len If points to 0, set to the number of irreps in the
 *                          corresponding row of \p lookup_tabl; Otherwise,
 *                          taken to indicate the length of \p prob_arr, such
 *                          additional elements in prob_len beyond the number of
 *                          irreps are zeroed.
 * \return sum of weights for all unoccupied orbitals before normalization
 */
double calc_u2_probs_half(hb_info *tens, double *prob_arr, uint8_t o1_orb,
                          uint8_t o2_orb, uint8_t u1_orb, uint8_t *det,
                          SymmInfo *symm, uint16_t *prob_len);


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
 * \param [in] symm     Object containing information about the symmetries of orbitals in the single-particle basis 
 * \return the weight for the excitation
 */
double calc_norm_wt(hb_info *tens, uint8_t *orbs, uint8_t *occ,
                    unsigned int n_elec, uint8_t *det,
                    SymmInfo *symm);


/*! \brief Perform multinomial sampling on the double excitation part of the
 *  Hamiltonian matrix using the heat-bath Power-Pitzer factorization
 *
 * \param [in] det          bit string representation of the origin determinant
 *                          defining the respective column of the Hamiltonian
 * \param [in] occ_orbs     list of occupied orbitals in det
 *                          (length \p num_elec)
 * \param [in] num_elec     number of occupied orbitals in det
 * \param [in] tens         hb_info struct containing the HB-PP parameters
 * \param [in] symm     Object containing information about the symmetries of orbitals in the single-particle basis
 * \param [in] num_sampl    number of double excitations to sample from the
 *                          column
 * \param [in] mt_obj       Reference to an initialized MT object for RN generation
 * \param [out] chosen_orbs Contains occupied (0th and 1st columns) and
 *                          unoccupied (2nd and 3rd columns) orbitals for each
 *                          excitation (dimensions num_sampl x 4)
 * \param [out] prob_vec    Contains probability for each excitation
 *                          (length num_sampl)
 * \return number of excitations sampled (less than num_sampl if some excitations
 *  were null)
 */
unsigned int hb_doub_multi(uint8_t *det, uint8_t *occ_orbs,
                           unsigned int num_elec, SymmInfo *symm,
                           hb_info *tens,
                           unsigned int num_sampl, std::mt19937 &mt_obj,
                           uint8_t (* chosen_orbs)[4], double *prob_vec);


/*! \brief A structure containing the various arrays needed to apply compression operations to a
 * hierarchical Hamiltonian factorization
 */
struct HBCompress {
    std::vector<double> vec1; ///< Vector elements before and after compression
    size_t vec_len; ///< Number of nonzero elements before and after compression
    
    // The following are used to index the values in each compression array
    // Before compression, elements are indexed by Slater determinants
    // After compression, elements are indexed by orbitals indicating single or double excitations from Slater determinants
    std::vector<size_t> det_indices1; ///< Before compression, indicates the index in another data structure of the Slater determinant index
    std::vector<size_t> det_indices2; ///< After compression, indicates the index in another data structure of the Slater determinant index
    uint8_t (*orb_indices1)[4]; ///< Orbital indices after compression: 0th index is 0 for double/1 for single, 1st is 1st occ, 2nd is 1st virt/2nd occ, 3rd is nothing/1st virt
    uint8_t (*orb_indices2)[4]; ///< Intermediate orbital indices
    
    // In hierarchical Hamiltonian factorizations, each element gives rise to a number of other elements upon multiplication
    // This array contains the number of elements in each of these "groups"
    std::vector<uint16_t> group_sizes;
    
    HBCompress(size_t length) : vec1(length), det_indices1(length), det_indices2(length), group_sizes(length) {
        orb_indices1 = (uint8_t (*)[4]) malloc(sizeof(uint8_t) * 4 * length);
        orb_indices2 = (uint8_t (*)[4]) malloc(sizeof(uint8_t) * 4 * length);
    }
    ~HBCompress() {
        free(orb_indices1);
        free(orb_indices2);
    }
};

/*! \brief When performing systematic compression, vector elements after each multiplication are represented by subweights:
 * Each element is represented as a product of an element in either \p vec1 or \p vec2 and a "subweight"
 */
struct HBCompressSys : HBCompress {
    std::vector<double> vec2; ///< Intermediate array to hold values after compressions
    
    Matrix<double> subwts; ///< Matrix of subweights. Row indices correspond to values in vec1/vec2, column indices correspond to subweight indices
    std::vector<uint32_t> ndiv; ///< If \p ndiv[idx] != 0, the subweights for idx are all equal, and the number of such subweights is \p ndiv[idx] . The corresponding row of \p subwts is not used.
    Matrix<bool> keep_sub; ///< Indicates which elements are to be preserved exactly in compression
    std::vector<double> wt_remain; ///< The sum of magnitudes of elements in each row that were not marked for exact preservation
    
    size_t (*comp_idx)[2]; ///< Used to index elements of nonzero elements after each compression. The first index in each row is the index in vec1/vec2, and the 2nd index is a subweight index.
    
    HBCompressSys(size_t length, size_t n_subwt) : vec2(length),
        subwts(length, n_subwt), ndiv(length), keep_sub(length, n_subwt),
        wt_remain(length),  HBCompress(length) {
        comp_idx = (size_t (*)[2]) malloc(sizeof(size_t) * 2 * length);
    }
    ~HBCompressSys() {
        free(comp_idx);
    }
};

/*! \brief When performing pivotal compression, multiplication by Hamiltonian matrix factor is performed explicitly.
 * No subweights. The vector is sequentially expanded into \p long_vec by a matrix multiplication and then compressed
 *  and transferred back to \p vec1
 */
struct HBCompressPiv : HBCompress {
    std::vector<double> long_vec;
    std::vector<bool> keep_idx; ///< Marks indices to be preserved exactly in compression
    std::vector<size_t> cmp_srt; ///< Scratch array used in compression
    
    HBCompressPiv(size_t length, size_t n_subwt) : long_vec(length * n_subwt),
        keep_idx(length * n_subwt), cmp_srt(length * n_subwt), HBCompress(length) {}
};


/*! \brief Apply the heat-bath power-pitzer Hamiltonian matrix factors to an initial vector, applying systematic compression after multiplication by each factor (except the last)
 *
 * \param [in] all_orbs     A matrix containing the occupied orbitals for each element in the initial vector
 * \param [in] all_dets     A matrix containing the bit-string representations of each determinant in the initial vector
 * \param [in, out] comp_scratch       Structure containing the intermediate vectors to use in the compression
 * \param [in] hb_probs     Structure containing the precomputed HB-PP probabilities
 * \param [in] symm     Structure containing the symmetry information about the one-particle basis
 * \param [in] p_doub      The probability of selecting a generic double excitation (defined in the HB-PP factorization)
 * \param [in] new_hb   Specifies whether to use the modified HB-PP factorization defined in Greene et al. (2020)
 * \param [in] mt_obj   For generating random numbers
 * \param [in] n_samp  Number of nonzero elements to keep in compression operations
 * \param [in] sing_mat_fxn     Returns the Hamiltonian matrix element for a single excitation (without parity)
 *                          given the orbitals involved in the excitation and the occupied orbitals in the origin determinant
 * \param [in] doub_mat_fxn     Returns the Hamiltonian matrix element for a double excitation (without parity)
 *                          given the orbitals involved in the excitation
 */
void apply_HBPP_sys(Matrix<uint8_t> &all_orbs, Matrix<uint8_t> &all_dets, HBCompressSys *comp_scratch,
                    hb_info *hb_probs, SymmInfo *symm, double p_doub, bool new_hb,
                    std::mt19937 &mt_obj, uint32_t n_samp,
                    std::function<double(uint8_t *, uint8_t *)> sing_mat_fxn,
                    std::function<double(uint8_t *)> doub_mat_fxn);


/*! \brief Apply the heat-bath power-pitzer Hamiltonian matrix factors to an initial vector, applying systematic compression after multiplication by each factor (except the last)
 *
 * \param [in] all_orbs     A matrix containing the occupied orbitals for each element in the initial vector
 * \param [in] all_dets     A matrix containing the bit-string representations of each determinant in the initial vector
 * \param [in, out] comp_scratch       Structure containing the intermediate vectors to use in the compression
 * \param [in] hb_probs     Structure containing the precomputed HB-PP probabilities
 * \param [in] symm     Structure containing the symmetry information about the one-particle basis
 * \param [in] p_doub      The probability of selecting a generic double excitation (defined in the HB-PP factorization)
 * \param [in] new_hb   Specifies whether to use the modified HB-PP factorization defined in Greene et al. (2020)
 * \param [in] mt_obj   For generating random numbers
 * \param [in] n_samp  Number of nonzero elements to keep in compression operations
 * \param [in] sing_mat_fxn     Returns the Hamiltonian matrix element for a single excitation (without parity)
 *                          given the orbitals involved in the excitation and the occupied orbitals in the origin determinant
 * \param [in] doub_mat_fxn     Returns the Hamiltonian matrix element for a double excitation (without parity)
 *                          given the orbitals involved in the excitation
 * \param [in] spin_parity      1 if targeting even-spin states with time-reveral symmetry, -1 for odd-spin states, 0 if time-reversal symmetry is not used
 */
void apply_HBPP_piv(Matrix<uint8_t> &all_orbs, Matrix<uint8_t> &all_dets, HBCompressPiv *comp_scratch,
                    hb_info *hb_probs, SymmInfo *symm, double p_doub, bool new_hb,
                    std::mt19937 &mt_obj, uint32_t n_samp,
                    std::function<double(uint8_t *, uint8_t *)> sing_mat_fxn,
                    std::function<double(uint8_t *)> doub_mat_fxn, int spin_parity);


#endif /* heat_bathPP_h */
