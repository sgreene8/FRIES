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
#include <stack>


/*! \brief Hash table used to to index Slater determinant indices in the
 * solution vector
 */
template <class el_type>
class HashTable {
    struct HashEntry {
        uint8_t *bit_string_;
        el_type value_;
        
        HashEntry(size_t idx_size) : value_(-1) {
            bit_string_ = (uint8_t *)malloc(idx_size * sizeof(uint8_t));
        };
        ~HashEntry() {
            free(bit_string_);
        }
    };
    
    std::vector<std::forward_list<HashEntry *>> buckets_; ///< Vector containing the hash entries for each bucket
    std::vector<uint32_t> scrambler_; ///< array of random integers to use for hashing
    uint8_t idx_size_; ///< Number of bytes used to encode each bit string index in the hash table
    std::stack<HashEntry *> recycler_;
    
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
        std::forward_list<HashEntry *> &row = buckets_[table_idx];
        
        uint32_t collisions = 0;
        el_type *ret_ptr = nullptr;
        for (HashEntry *entry : row) {
            if (!memcmp(det, entry->bit_string_, idx_size_)) {
                ret_ptr = &entry->value_;
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
            HashEntry *new_entry;
            if (recycler_.empty()) {
                new_entry = new HashEntry(idx_size_);
            }
            else {
                new_entry = recycler_.top();
                recycler_.pop();
                new_entry->value_ = -1;
            }
            std::copy(det, det + idx_size_, new_entry->bit_string_);
            row.push_front(new_entry);
            ret_ptr = &new_entry->value_;
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
            uint16_t collisions = 0;
            std::forward_list<HashEntry *> row = buckets_[table_idx];
            for (HashEntry *entry : row) {
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
        std::forward_list<HashEntry *> &row = buckets_[table_idx];
        if (!row.empty()) {
            auto it1 = row.begin();
            if (!memcmp(det, (*it1)->bit_string_, idx_size_)) {
                recycler_.push(*it1);
                row.pop_front();
            }
            else {
                auto it2 = it1;
                it2++;
                while (it2 != row.end()) {
                    if (!memcmp((*it2)->bit_string_, det, idx_size_)) {
                        recycler_.push(*it2);
                        row.erase_after(it1);
                        return;
                    }
                    it2++;
                    it1++;
                }
            }
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
