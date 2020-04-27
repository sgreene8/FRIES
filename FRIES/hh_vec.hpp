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
              unsigned int n_elec, int n_procs, std::function<double(const uint8_t *)> diag_fxn, uint8_t n_vecs): DistVec<el_type>(size, add_size, rn_ptr, n_sites * 2 + n_sites * max_ph, n_elec, n_procs, diag_fxn, NULL, n_vecs), neighb_(size, 2 * (n_elec + 1)), n_sites_(n_sites), ph_bits_(max_ph), phonon_nums_(size, n_sites_) {
        
    }
    
    
    /*! \brief Generate list of occupied orbitals from bit-string representation of
     *  a determinant
     *
     * This implementation uses the procedure in Sec. 3.1 of Booth et al. (2014)
     * \param [in] det          bit string to be parsed
     * \param [out] occ_orbs    Occupied orbitals in the determinant
     * \return number of 1 bits in the bit string
     */
    uint8_t gen_orb_list(uint8_t *det, uint8_t *occ_orbs) {
        unsigned int byte_idx, elec_idx;
        uint8_t n_elec, det_byte, bit_idx;
        elec_idx = 0;
        uint8_t tot_elec = 0;
        uint8_t max_byte = CEILING(n_sites_ * 2, 8);
        byte_table *table = DistVec<el_type>::tabl_;
        for (byte_idx = 0; byte_idx < max_byte; byte_idx++) {
            det_byte = det[byte_idx];
            if ((n_sites_ * 2 % 8) > 0 && (byte_idx == max_byte - 1)) {
                det_byte &= (1 << (n_sites_ * 2 % 8)) - 1;
            }
            n_elec = table->nums[det_byte];
            for (bit_idx = 0; bit_idx < n_elec; bit_idx++) {
                occ_orbs[elec_idx + bit_idx] = (8 * byte_idx + table->pos[det_byte][bit_idx]);
            }
            elec_idx = elec_idx + n_elec;
            tot_elec += n_elec;
        }
        
        return tot_elec;
    }
    
    
    /*! \brief Hash function mapping vector index to MPI process
     *
     * \param [in] idx          Vector index
     * \return process index from hash value
     */
    int idx_to_proc(uint8_t *idx) {
        unsigned int n_elec = (unsigned int)DistVec<el_type>::occ_orbs_.cols();
        uint8_t orbs[n_elec];
        gen_orb_list(idx, orbs);
        uint8_t phonons[n_sites_];
        decode_phonons(idx, phonons);
        unsigned long long hash_val = hash_fxn(orbs, n_elec, phonons, n_sites_, DistVec<el_type>::proc_scrambler_);
        int n_procs = 1;
#ifdef USE_MPI
        MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
#endif
        return hash_val % n_procs;
    }

    /*! \brief Hash function mapping vector index to local hash value
     *
     * The local hash value is used to find the index on a particular processor
     *
     * \param [in] idx          Vector index
     * \return hash value
     */
    uintmax_t idx_to_hash(uint8_t *idx, uint8_t *orbs) {
        unsigned int n_elec = (unsigned int)DistVec<el_type>::occ_orbs_.cols();
        if (gen_orb_list(idx, orbs) != n_elec) {
            uint8_t n_bytes = DistVec<el_type>::indices_.cols();
            char det_txt[n_bytes * 2 + 1];
            print_str(idx, n_bytes, det_txt);
            fprintf(stderr, "Error: determinant %s created with an incorrect number of electrons.\n", det_txt);
        }
        uint8_t phonons[n_sites_];
        decode_phonons(idx, phonons);
        return hash_fxn(orbs, n_elec, phonons, n_sites_, DistVec<el_type>::vec_hash_->scrambler);
    }
    
    /*! \brief Double the maximum number of elements that can be stored */
    void expand() {
        DistVec<el_type>::expand();
        size_t new_size = DistVec<el_type>::max_size_;
        neighb_.reshape(new_size, neighb_.cols());
        phonon_nums_.reshape(new_size, phonon_nums_.cols());
    }
    
    /*! \returns A reference to the Matrix used to store information about empty neighboring orbitals of
     *              seach determinant in the Hubbard model*/
    Matrix<uint8_t> &neighb(){
        return neighb_;
    }
    
    /*! \return A reference to the Matrix used to store the quantum numbers of each phonon for each state index in the vector
     */
    Matrix<uint8_t> &phonon_nums() {
        return phonon_nums_;
    }

    /*! \brief Get a pointer to the list of phonon numbers for a basis state in the vector
     *
     * \param [in] pos          The row index of the basis state in the \p indices_  matrix
     */
    uint8_t *phonons_at_pos(size_t pos) {
        return phonon_nums_[pos];
    }
    
    /*! \brief Calculate the total number of phonons at all sites for a basis element
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
        if (((2 * n_sites_ - 1) % 8) != 0) {
            neib_bits[n_bytes - 1] &= (1 << ((2 * n_sites_ - 1) % 8)) - 1;
        }
        
        neighbors[0] = find_bits(neib_bits, &neighbors[1], n_bytes, DistVec<el_type>::tabl_);
        
        mask = ~det[0] << 1;
        neib_bits[0] = det[0] & mask;
        for (byte_idx = 1; byte_idx < n_bytes; byte_idx++) {
            mask = ~det[byte_idx] << 1;
            mask |= (~det[byte_idx - 1] >> 7) & 1;
            neib_bits[byte_idx] = det[byte_idx] & mask;
        }
        zero_bit(neib_bits, n_sites_); // open boundary conditions
        if (((2 * n_sites_) % 8) != 0) {
            neib_bits[n_bytes - 1] &= (1 << ((2 * n_sites_) % 8)) - 1;
        }
        
        neighbors[n_elec + 1] = find_bits(neib_bits, &neighbors[n_elec + 1 + 1], n_bytes, DistVec<el_type>::tabl_);
    }
    
    
    /*! \brief Calculate the phonon quantum numbers from a bit-string representation of a determinant
     *
     * Calculates the phonon quanta at each lattice site
     *
     * \param [in] det      Bit-string representation of determinant
     * \param [out] numbers     Array in which to store the decoded numbers
     */
    void decode_phonons(uint8_t *det, uint8_t *numbers) {
        uint8_t mask = (1 << ph_bits_) - 1;
        size_t curr_bit;
        size_t curr_byte;
        for (uint8_t site_idx = 0; site_idx < n_sites_; site_idx++) {
            curr_bit = 2 * n_sites_ + site_idx * ph_bits_;
            curr_byte = curr_bit / 8;
            numbers[site_idx] = (det[curr_byte] >> (curr_bit % 8)) & mask;
            if ((curr_bit % 8) + ph_bits_ > 8) {
                numbers[site_idx] += (det[curr_byte + 1] << (8 - (curr_bit % 8))) & mask;
            }
        }
    }
    
    /*! \brief Generate a new determinant by changing the phonon number at one position
     *
     * \param [in] orig     Original determinant from which to create new determinant
     * \param [out] new_det     Upon return, contains the bit-string representation of the new determinant
     * \param [in] site_idx       Site index of the phonon number to be incremented
     * \param [in] change       +1 if phonon number should be increased, -1 if should be decreased
     * \return 1 if new determinant was created successfully, 0 if not
     */
    int det_from_ph(uint8_t *orig, uint8_t *new_det, uint8_t site_idx, int change) {
        uint8_t bit_idx = 2 * n_sites_ + site_idx * ph_bits_;
        uint16_t det_segment = orig[bit_idx / 8];
        uint8_t n_bytes = CEILING(DistVec<el_type>::n_bits_, 8);
        if (bit_idx / 8 + 1 < n_bytes) {
            det_segment |= orig[bit_idx / 8 + 1] << 8;
        }
        uint16_t mask = (1 << ((bit_idx % 8) + ph_bits_)) - (1 << (bit_idx % 8));
        uint16_t ph_num = (det_segment & mask) >> (bit_idx % 8);
        if (change == 1 && ph_num == ((1 << ph_bits_) - 1)) {
            fprintf(stderr, "Warning: max phonon number reached\n");
            return 0;
        }
        if (change == -1 && ph_num == 0) {
            return 0;
        }
        ph_num += change;
        ph_num <<= (bit_idx % 8);
        det_segment &= ~mask;
        det_segment |= ph_num;
        memcpy(new_det, orig, n_bytes);
        new_det[bit_idx / 8] = det_segment & 255;
        if (bit_idx / 8 + 1 < n_bytes) {
            new_det[bit_idx / 8 + 1] = det_segment >> 8;
        }
        return 1;
    }
    
    
    /*! \brief Calculate the sum of all phonon quantum numbers for a basis element
     * \param [in] idx      Index of the basis element in the vector
     */
    unsigned int total_ph(size_t idx) {
        unsigned int sum = 0;
        uint8_t *ph_row = phonon_nums_[idx];
        for (size_t ph_idx = 0; ph_idx < n_sites_; ph_idx++) {
            sum += ph_row[ph_idx];
        }
        return sum;
    }
    
    
    void initialize_at_pos(size_t pos, uint8_t *orbs) {
        DistVec<el_type>::initialize_at_pos(pos, orbs);
        uint8_t *det = DistVec<el_type>::indices_[pos];
        find_neighbors_1D(det, neighb_[pos]);
        decode_phonons(det, phonon_nums_[pos]);
    }
};

#endif /* hh_vec_h */
