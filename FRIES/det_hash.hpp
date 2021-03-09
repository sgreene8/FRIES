/*! \file Class definition for a hash table with specific capabilities for bit strings
 */

#ifndef det_hash_hpp
#define det_hash_hpp

#include <vector>
#include <FRIES/det_store.h>
#include <forward_list>
#include <iostream>
#include <fstream>
#include <functional>
#include <mpi.h>
#include <sstream>


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
    };
    
    std::vector<std::forward_list<hash_entry>> buckets_; ///< Vector of pointers to linked list of hash_entry structs at each positiion
    std::vector<uint32_t> scrambler_; ///< array of random integers to use for hashing
    uint8_t idx_size_; ///< Number of bytes used to encode each bit string index in the hash table
public:

    /*! \brief Constructor for the hash table
     *
     * \param [in] table_size   Desired length of the hash table
     * \param [in] rand_ints    Vector of random integers to use for hashing (length must be >= # of spin orbitals in basis)
     */
    HashTable(size_t table_size, const std::vector<uint32_t> rand_ints) : buckets_(table_size), scrambler_(rand_ints), idx_size_(CEILING(rand_ints.size(), 8)) {}
    

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
        std::forward_list<hash_entry> &list = buckets_[table_idx];
        
        unsigned int collisions = 0;
        el_type *ret_ptr = nullptr;
        for (hash_entry &entry : list) {
            if (!memcmp(det, entry.det, idx_size_)) {
                ret_ptr = &entry.val;
                break;
            }
            collisions++;
        }
        
        if (collisions > 20) {
            char det_str[2 * idx_size_];
            print_str(det, idx_size_, det_str);
            std::cerr << "Line " << table_idx << " in the hash table has >20 hash collisions (det: " << det_str << ", hash: " << hash_val << ", " << (ret_ptr ? "found" : "not found") << ")\n";
        }
        if (!ret_ptr && create) {
            list.emplace_front();
            hash_entry &entry = list.front();
            entry.det = det;
            entry.val = -1;
            ret_ptr = &entry.val;
        }
        return ret_ptr;
    }

    /*! \brief Print number of elements in each row of hash table
     */
    void print_ht() {
        int my_rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        std::stringstream buffer;
        buffer << "hash" << my_rank << ".txt";
        std::ofstream file_p(buffer.str());
        for (size_t table_idx = 0; table_idx < buckets_.size();
             table_idx++) {
            unsigned int collisions = 0;
            std::forward_list<hash_entry> &list = buckets_[table_idx];
            for (hash_entry &entry : list) {
                collisions++;
            }
            file_p << collisions << "\n";
        }
        file_p.close();
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
        std::forward_list<hash_entry> &list = buckets_[table_idx];
        uint8_t local_size = idx_size_;
        std::function<bool(const hash_entry&)> pred = [det, local_size](const hash_entry& value) {
            return !memcmp(value.det, det, local_size);
        };
        list.remove_if(pred);
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
