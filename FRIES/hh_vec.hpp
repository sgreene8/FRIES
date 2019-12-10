/*! \file Class definition for a subclass of SparseVector used for Hubbard-Holstein problems
 
 */

#ifndef hh_vec_h
#define hh_vec_h

#include <FRIES/vec_utils.hpp>

template <class el_type>
class HubHolVec : public DistVec<el_type> {
private:
    Matrix<uint8_t> neighb_; ///< Pointer to array containing information about empty neighboring orbitals for Hubbard model
    uint8_t n_sites_; ///< Number of sites along one dimension of the lattice
    uint8_t ph_bits_; ///< Number of bits used to encode the occupancy number for each phonon
    Matrix<uint8_t> phonon_nums_; ///< Phonon quantum numbers for each state
public:
    /*! \brief Constructor for HubHolVec object
     * \param [in] size         Maximum number of elements to be stored in the vector
     * \param [in] add_size     Maximum number of elements per processor to use in Adder object
     * \param [in] rn_ptr       Pointer to an mt_struct object for RN generation
     * \param [in] n_sites        Number of sites along one dimension of the lattice
     * \param [in] max_ph       Number of bits used initially to encode the occupancy number for each phonon
     * \param [in] n_elec       Number of electrons represented in each vector index
     * \param [in] n_procs Number of MPI processes over which to distribute vector elements
     */
    HubHolVec(size_t size, size_t add_size, mt_struct *rn_ptr, uint8_t n_sites, uint8_t max_ph,
              unsigned int n_elec, int n_procs): DistVec<el_type>(size, add_size, rn_ptr, n_sites * 2 + n_sites * max_ph, n_elec, n_procs), neighb_(size, 2 * (n_elec + 1)), n_sites_(n_sites), ph_bits_(max_ph), phonon_nums_(size, n_sites_) {
        
    }
    
    /*! \brief Double the maximum number of elements that can be stored */
    void expand() {
        DistVec<el_type>::expand();
        neighb_.reshape(DistVec<el_type>::max_size_, neighb_.cols());
    }
    
    /*! \returns A reference to the Matrix used to store information about empty neighboring orbitals of
     *              seach determinant in the Hubbard model*/
    Matrix<uint8_t> &neighb(){
        return neighb_;
    }
    
    /*! \return A reference to the Matrix used to store the quantum numbers of each phonon for each state index in the vector
     */
    Matrix<uint8_t> *phonon_nums() {
        return phonon_nums_;
    }
    
    /*! \brief Calculate the total number of phonons at all sizes for a basis element
     * \param [in] idx       Index of the basis element in the DistVector object
     * \return Number of phonons
     */
    uint8_t tot_ph_at_idx(size_t idx) {
        uint8_t phonons = 0;
        for (size_t ph_idx = 0; ph_idx < n_sites_; ph_idx++) {
            phonons += phonon_nums_(idx, ph_idx);
        }
        return phonons;
    }

    /*! \brief Generate lists of occupied orbitals in a determinant that are
     *  adjacent to an empty orbital if the orbitals represent sites on a 1-D lattice
     *
     * \param [in] det          bit string representation of the determinant
     * \param [out] neighbors   2-D array whose 0th row indicates orbitals with an
     *                          empty adjacent orbital to the left, and 1st row is
     *                          for empty orbitals to the right. Elements in the 0th
     *                          column indicate number of elements in each row.
     */
    void find_neighbors_1D(uint8_t *det, uint8_t *neighbors) {
        size_t n_elec = DistVec<el_type>::occ_orbs_.cols();
        size_t byte_idx = 0;
        size_t n_bytes = CEILING(n_sites_ * 2, 8);
        uint8_t neib_bits[n_bytes];
        
        uint8_t mask = det[0] >> 1;
        for (byte_idx = 1; byte_idx < n_bytes; byte_idx++) {
            mask |= (det[byte_idx] & 1) << 7;
            neib_bits[byte_idx - 1] = det[byte_idx - 1] & ~mask;
            
            mask = det[byte_idx] >> 1;
        }
        neib_bits[n_bytes - 1] = det[n_bytes - 1] & ~mask;
        
        zero_bit(neib_bits, n_sites_ - 1);
        zero_bit(neib_bits, 2 * n_sites_ - 1); // open boundary conditions
        
        neighbors[0] = DistVec<el_type>::gen_orb_list(neib_bits, &neighbors[1]);
        
        mask = ~det[0] << 1;
        neib_bits[0] = det[0] & mask;
        for (byte_idx = 1; byte_idx < n_bytes; byte_idx++) {
            mask = ~det[byte_idx] << 1;
            mask |= (~det[byte_idx - 1] >> 7) & 1;
            neib_bits[byte_idx] = det[byte_idx] & mask;
        }
        zero_bit(neib_bits, n_sites_); // open boundary conditions
        
        neighbors[n_elec + 1] = DistVec<el_type>::gen_orb_list(neib_bits, &neighbors[n_elec + 1 + 1]);
    }
    
    
    void decode_phonons(uint8_t *det, uint8_t *numbers) {
        uint8_t mask = (1 << ph_bits_) - 1;
        size_t curr_bit;
        size_t curr_byte;
        for (uint8_t site_idx = 0; site_idx < n_sites_; site_idx++) {
            curr_bit = 2 * n_sites_ + site_idx * ph_bits_;
            curr_byte = curr_bit / 8;
            numbers[site_idx] = (det[curr_byte] >> (curr_bit % 8)) & mask;
            if ((curr_bit % 8) + ph_bits_ > 8) {
                numbers[site_idx] += det[curr_byte + 1] << (8 - (curr_bit % 8));
            }
        }
    }
    
    /*! \brief Generate a new determinant by increasing the phonon number at one position
     *
     * \param [in] orig     Original determinant from which to create new determinant
     * \param [out] new_det     Upon return, contains the bit-string representation of the new determinant
     * \param [in] site_idx       Site index of the phonon number to be incremented
     * \return 1 if new determinant was created successfully, 0 if not
     */
    int det_by_inc_ph(uint8_t *orig, uint8_t *new_det, uint8_t site_idx) {
        uint8_t bit_idx = 2 * n_sites_ + site_idx * ph_bits_;
        uint8_t mask1 = (1 << ((bit_idx % 8) + ph_bits_)) - (1 << (bit_idx % 8));
        uint8_t ph_num = orig[bit_idx / 8] >> (bit_idx % 8);
        if ((bit_idx % 8) + ph_bits_ > 8) {
            ph_num |= orig[bit_idx / 8 + 1] << (8 - (bit_idx % 8));
        }
        ph_num++;
        memcpy(new_det, orig, CEILING(DistVec<el_type>::n_bits_, 8));
        new_det[bit_idx / 8] &= ~(((1 << ph_bits_) - 1) << (bit_idx % 8));
        new_det[bit_idx / 8] |= ph_num << (bit_idx % 8);
        if ((bit_idx % 8) + ph_bits_ > 8) {
//            new_det[bit_idx / 8 + 1] &= ~(((1 << ())
        }
    }
    
    
    void initialize_at_pos(size_t pos) {
        DistVec<el_type>::initialize_at_pos(pos);
        uint8_t *det = DistVec<el_type>::indices_[pos];
        find_neighbors_1D(det, neighb_[pos]);
        decode_phonons(det, phonon_nums_[pos]);
    }
};

#endif /* hh_vec_h */
