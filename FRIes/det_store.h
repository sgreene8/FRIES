/*! \file Utilities for keeping track of Slater determinant indices of a sparse
 * vector.
 */

#ifndef det_store_h
#define det_store_h

#include <stdio.h>
#include <stdlib.h>
#include <FRIes/Ext_Libs/dcmt/dc.h>


/*! \brief Stack data structure used to keep track of unused indices in a
 * continuous array of sparse vector indices
 */
typedef struct {
    size_t *storage; ///< Array used to store the contents of the stack
    size_t buf_size; ///< Size of the \p storage array
    size_t curr_idx; ///< Current number of elements in the stack
} stack_s;


/*! \brief Allocate memory and initialize variables within a stack structure.
 *
 * \param [in] buf_size     Maximum number of elements that can be stored in the
 *                          stack.
 * \return pointer to newly allocated stack structure
 */
stack_s *setup_stack(size_t buf_size);


/*! \brief Insert element into stack
 *
 * \param [in] buf          Stack structure in which to insert
 * \param [in] val          Element to be written
 */
void push(stack_s *buf, size_t val);


/*! \brief Read element from stack
 *
 * \param [in] buf          Stack structure to be read from
 * \return value read from stack, or -1 if stack is empty
 */
ssize_t pop(stack_s *buf);


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


#endif /* det_store_h */
