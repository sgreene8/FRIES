/*! \file
 *
 * \brief Utilities for keeping track of Slater determinant indices of a
 * sparse vector.
 */

#ifndef det_store_h
#define det_store_h

#include <stdio.h>
#include <stdlib.h>
#include <FRIES/Ext_Libs/dcmt/dc.h>


#ifdef __cplusplus
extern "C" {
#endif

/*! \brief An element in a singly-linked list as may be used to represent a stack
 */
struct stack_entry {
    size_t idx; ///<The data associated with the element
    struct stack_entry *next; ///< A pointer to the next element in the list, or NULL if last element
};
typedef struct stack_entry stack_entry;


/*! \brief Data structure used to store entries in the hash table
 *
 * The keys in the hash table are the bit-string representations of Slater
 * determinants, and the values are the indices in the array representing the
 * sparse vector, or -1 if the key has yet to be assigned an index
 */
struct hash_entry {
    long long det; ///< key for the hash table in bit-string form
    ssize_t val; ///< index in main determinant array, or -1 if uninitialized
    struct hash_entry *next; // pointer to next entry in linked list
};
typedef struct hash_entry hash_entry;


/*! \brief Hash table used to to index Slater determinant indices in the
 * solution vector
 */
typedef struct {
    hash_entry *recycle_list; ///< Linked list of hash_entry objects for reuse
    size_t length; ///< Length of the hash table
    hash_entry **buckets; ///< Array (length \p length) of pointers to linked list of hash_entry objects at each position
    unsigned int *scrambler; // array of random integers to use for hashing
} hash_table;


/*! \brief Set up the hash table
 *
 * \param [in] table_size   Desired length of the hash table
 * \param [in] rn_gen       Pointer to a MT object to use for generating random
 *                          intgers
 * \param [in] rn_len       Number of random integers to use in the hash
 *                          function (must be >= # of spin orbitals in basis)
 * \return pointer to newly allocated hash_table
 */
hash_table *setup_ht(size_t table_size, mt_struct *rn_gen, unsigned int rn_len);


/*! \brief Read value from hash table
 *
 * \param [in] table        pointer to hash table struct
 * \param [in] det          bit-string representation of determinant (key for
 *                          the hash table)
 * \param [in] hash_val     hash value for the determinant, calculated using
 *                          hash_fxn
 * \param [in] create       if nonzero, create a new entry in the hash table
 * \return pointer to value in hash table, or NULL if key not found and create
 * is 0
 */
ssize_t *read_ht(hash_table *table, long long det, unsigned long long hash_val,
                 int create);


/*! \brief Delete value in hash table. If not found, do nothing
 *
 * \param [in] table        pointer to hash table struct
 * \param [in] det          bit-string representation of determinant (key for
 *                          the hash table)
 * \param [in] hash_val     hash value for the determinant, calculated using
 *                          hash_fxn
 */
void del_ht(hash_table *table, long long det, unsigned long long hash_val);


/*! \brief Hash function for Slater determinants in bit-string representation
 *
 * Calculates the Merkle-Damgard hash value from occupied orbitals in
 * determinant, according to Algorithm 1 in Booth et al. (2014)
 * \param [in] occ_orbs     list of occupied orbitals in the determinant
 *                          (length \p n_elec)
 * \param [in] n_elec       Number of occupied orbitals in the determinant
 * \param [in] rand_nums    List of random integers used in hash function, must
 *                          have at least (max(occ_orbs) + 1) elements
 * \return the calculated hash value
 */
unsigned long long hash_fxn(unsigned char *occ_orbs, unsigned int n_elec, unsigned int *rand_nums);


#ifdef __cplusplus
}
#endif

#endif /* det_store_h */
