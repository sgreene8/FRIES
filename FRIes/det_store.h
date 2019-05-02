//
//  det_store.h
//  FRIes
//
//  Created by Samuel Greene on 4/14/19.
//  Copyright © 2019 Samuel Greene. All rights reserved.
//

#ifndef det_store_h
#define det_store_h

#include <stdio.h>
#include <stdlib.h>
#include "dc.h"

#define MaxBuckets 5

struct stack {
    size_t *storage;
    size_t buf_size;
    size_t curr_idx;
};
typedef struct stack stack_s;


struct hash_table {
    size_t length;
    ssize_t **bucket_vals; // indices in main determinant array, or -1 if uninitialized
    long long **bucket_dets; // keys for the hash table in bit-string form
    unsigned int *bucket_sizes;
    unsigned int *scrambler; // array of random integers to use for hashing
};
typedef struct hash_table hash_table;


/* Allocate memory for hash table and initialize elements
 
 Arguments
 ---------
 table_size: number of elements in the hash table
 rn_gen: pointer to a MT object to use for generating random intgers
 rn_len: number of random integers to use for hashing (>= # of spin orbitals in basis)
 
 Returns
 -------
 pointer to newly allocated table
 */
hash_table *setup_ht(size_t table_size, mt_struct *rn_gen, unsigned int rn_len);


/* Return reference to memory location in hash table corresponding to a determinant.
 
 Arguments
 ---------
 table: pointer to hash table struct
 det: bit-string representation of determinant
 hash_val: hash value for the determinant, calculated using hash_fxn
 create: if 1, create a new entry in the hash table
 
 Returns
 -------
 pointer to memory location, or NULL if none found
 */
ssize_t *read_ht(hash_table *table, long long det, unsigned long long hash_val,
                 int create);

/* Delete value in HT. If not found, do nothing.
 
 Arguments
 ---------
 (see read_ht)
 */
void del_ht(hash_table *table, long long det, unsigned long long hash_val);

/* Calculate Merkle-Damgard hash value from occupied orbitals in determinant,
 according to Algorithm 1 in Booth et al. (2014)
 
 Arguments
 ---------
 occ_orbs: list of occupied orbitals
 n_elec: number of elements in occ_orbs
 rand_nums: list of random integers, must have at least (max(occ_orbs) + 1) elements
 
 Returns
 -------
 hash value
 */
unsigned long long hash_fxn(unsigned char *occ_orbs, unsigned int n_elec, unsigned int *rand_nums);


/* Insert element into stack
 
 Arguments
 ---------
 buf: stack structure to be written to
 val: element to be written
 */
void push(stack_s *buf, size_t val);


/* Read element from stack
 
 Arguments
 ---------
 buf: stack structure to be read from
 
 Returns
 -------
 value read from buffer, or -1 if stack is empty
 */
ssize_t pop(stack_s *buf);


/* Allocate memory and initialize variables within a stack structure.
 
 Arguments
 ---------
 size: Maximum number of elements that can be stored in the stack.
 
 Returns
 -------
 pointer to newly allocated structure.
 */
stack_s *setup_stack(size_t buf_size);


#endif /* det_store_h */
