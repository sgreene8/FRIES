/*! \file Class definition for a hash table with specific capabilities for bit strings
 */

#ifndef det_hash_hpp
#define det_hash_hpp

#include <stack>
#include <vector>
#include <FRIES/det_store.h>


/*! \brief Hash table used to to index Slater determinant indices in the
 * solution vector
 */
template <class el_type>
class HashTable {
    /*! \brief Data structure used to store entries in the hash table
     *
     * The keys in the hash table are the bit-string representations of Slater
     * determinants, and the values are the indices in the array representing the
     * sparse vector, or -1 if the key has yet to be assigned an index
     */
    struct hash_entry {
        uint8_t *det; ///< key for the hash table in bit-string form
        el_type val; ///< index in main determinant array, or -1 if uninitialized
        struct hash_entry *next; ///< pointer to next entry in linked list
    };
    
    std::stack<hash_entry *> recycler_; ///< Stack containing hash_entry structs for reuse
    std::vector<hash_entry *> buckets_; ///< Vector of pointers to linked list of hash_entry structs at each positiion
    std::vector<unsigned int> scrambler_; ///< array of random integers to use for hashing
    uint8_t idx_size_; ///< Number of bytes used to encode each bit string index in the hash table
public:

    /*! \brief Constructor for the hash table
     *
     * \param [in] table_size   Desired length of the hash table
     * \param [in] rn_gen       Pointer to a MT object to use for generating random
     *                          integers
     * \param [in] rn_len       Number of random integers to use in the hash
     *                          function (must be >= # of spin orbitals in basis)
     */
    HashTable(size_t table_size, mt_struct *rn_gen, unsigned int rn_len) : buckets_(table_size, nullptr), scrambler_(rn_len), idx_size_(CEILING(rn_len, 8)) {};
    

    /*! \brief Read value from hash table
     *
     * \param [in] det          bit-string representation of determinant (key for
     *                          the hash table)
     * \param [in] hash_val     hash value for the determinant, calculated using
     *                          hash_fxn
     * \param [in] create       if true, create a new entry in the hash table
     * \return pointer to value in hash table, or NULL if key not found and create
     * is false
     */
    el_type *read(uint8_t *det, uintmax_t hash_val, bool create){
        size_t table_idx = hash_val % buckets_.size();
        // address of location storing address of next entry
        hash_entry **prev_ptr = &(buckets_[table_idx]);
        // address of next entry
        hash_entry *next_ptr = *prev_ptr;
        unsigned int collisions = 0;
        while (next_ptr) {
            if (bit_str_equ(det, next_ptr->det, idx_size_)) {
                break;
            }
            collisions++;
            prev_ptr = &(next_ptr->next);
            next_ptr = *prev_ptr;
        }
        if (collisions > 20) {
            fprintf(stderr, "There is a line in the hash table with >20 hash collisions.\n");
        }
        if (next_ptr) {
            return &(next_ptr->val);
        }
        else if (create) {
            if (!recycler_.empty()) {
                next_ptr = recycler_.top();
                recycler_.pop();
            }
            else {
                next_ptr = (hash_entry *) malloc(sizeof(hash_entry));
                next_ptr->det = (uint8_t *) malloc(idx_size_);
            }
            *prev_ptr = next_ptr;
            memcpy(next_ptr->det, det, idx_size_);
            next_ptr->next = NULL;
            next_ptr->val = -1;
            return &(next_ptr->val);
        }
        else
            return NULL;
    }
    

    /*! \brief Delete value in hash table. If not found, do nothing
     *
     * \param [in] det          bit-string representation of determinant (key for
     *                          the hash table)
     * \param [in] hash_val     hash value for the determinant, calculated using
     *                          hash_fxn
     */
    void del_entry(uint8_t *det, uintmax_t hash_val) {
        size_t table_idx = hash_val % buckets_.size();
        // address of location storing address of next entry
        hash_entry **prev_ptr = &(buckets_[table_idx]);
        // address of next entry
        hash_entry *next_ptr = *prev_ptr;
        while (next_ptr) {
            if (bit_str_equ(next_ptr->det, det, idx_size_)) {
                break;
            }
            prev_ptr = &(next_ptr->next);
            next_ptr = *prev_ptr;
        }
        if (next_ptr) {
            *prev_ptr = next_ptr->next;
            recycler_.push(next_ptr);
        }
    }
    

    /*! \brief Hash function for Slater determinants in bit-string representation
     *
     * Calculates the Merkle-Damgard hash value from occupied orbitals in
     * determinant, according to Algorithm 1 in Booth et al. (2014)
     * \param [in] occ_orbs     list of occupied orbitals in the determinant
     *                          (length \p n_elec)
     * \param [in] n_elec       Number of occupied orbitals in the determinant
     *                          have at least (max(occ_orbs) + 1) elements
     * \return the calculated hash value
     */
    uintmax_t hash_fxn(uint8_t *occ_orbs, unsigned int n_elec, uint8_t *phonon_nums, unsigned int n_phonon) {
        uintmax_t hash = 0;
        unsigned int i;
        for (i = 0; i < n_elec; i++) {
            hash = 1099511628211LL * hash + (i + 1) * scrambler_[occ_orbs[i]];
        }
        for (i = 0; i < n_phonon; i++) {
            hash = 1099511628211LL * hash + (i + 1) * scrambler_[phonon_nums[i]];
        }
        return hash;
    }
};

#endif /* det_hash_hpp */
