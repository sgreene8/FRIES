/*! \file
 *
 * \brief Utilities for storing and manipulating sparse vectors
 *
 * Supports sparse vectors distributed among multiple processes if USE_MPI is
 * defined
 */

#ifndef vec_utils_h
#define vec_utils_h

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "det_store.h"
#include "Hamiltonians/hub_holstein.h"
#include "Ext_Libs/dcmt/dc.h"
#include "mpi_switch.h"


/*! \brief Struct used to add elements to sparse vector
 *
 * Elements to be added are buffered in send buffers until perform_add() is
 * called, at which point they are distributed into receive buffers located in
 * their corresponding MPI processes
 */
typedef struct{
    dtype type; ///< Type of elements to be added
    size_t size; ///< Maximum number of elements per MPI process in send and receive buffers
    long long *send_idx; ///< Send buffer for element indices
    long long *recv_idx; ///< Receive buffer for element indices
    void *send_vals; ///< Send buffer for element values
    void *recv_vals; ///< Receive buffer for element values
    int *send_cts; ///< Number of elements in the send buffer for each process
    int *recv_cts; ///< Number of elements in the receive buffer for each process
    int *displacements; ///< Array positions in buffers corresponding to each process
} adder;


/*! \brief Setup an adder struct
 *
 * \param [in] size         Maximum number of elements per MPI process to use in
 *                          buffers
 * \param [in] type         Type of elements to be added
 * \return pointer to newly created adder struct
 */
adder *init_adder(size_t size, dtype type);


/*! \brief Struct for storing a sparse vector */
typedef struct {
    long long *indices; ///< Array of indices of vector elements
    void *values; ///< Array of values of vector elements
    double *matr_el; ///< Array of pre-calculated diagonal matrix elements associated with each vector element
    size_t max_size; ///< Maximum number of vector elements that can be stored
    size_t curr_size; ///< Current number of vector elements stored
    hash_table *vec_hash; ///< Hash table for quickly finding indices in the \p indices array
    stack_s *vec_stack; ///< Pointer to stack struct for managing available positions in the indices array
    unsigned int *proc_scrambler; ///< Array of random numbers used in the hash function for assigning vector indices to MPI processes
    byte_table *tabl; ///< Struct used to decompose determinant indices into lists of occupied orbitals
    unsigned int n_elec; ///< Number of electrons represented by determinant bit-string indices
    unsigned char *occ_orbs; ///< 2-D array containing lists of occupied orbitals for each determniant index (dimensions \p max_size x \p n_elec)
    unsigned char *neighb; ///< Pointer to array containing information about empty neighboring orbitals for Hubbard model
    unsigned int n_sites; ///< Number of sites along one dimension of the Hubbard lattice, if applicable
    dtype type; ///< Type of elements in vector
    adder *my_adder; ///< Pointer to adder struct for buffered addition of elements distributed across MPI processes
    int n_nonz; /// Current number of nonzero elements in vector
} dist_vec;


/*! \brief Set up a dist_vec struct
 *
 * \param [in] size         Maximum number of elements to be stored
 * \param [in] add_size     Maximum number of elements per processor to use in
 *                          adder buffers
 * \param [in] rn_ptr       Pointer to an mt_struct object for RN generation
 * \param [in] n_orb        Number of spatial orbitals in the basis (half the
 *                          length of the vector of random numbers for the
 *                          hash function for processors)
 * \param [in] n_elec       Number of electrons represented in each vector index
 * \param [in] vec_type     Data type of vector elements
 * \param [in] n_sites      Number of sites along one dimension of the Hubbard
 *                          lattice, if applicable
 * \return pointer to the newly allocated struct
 */
dist_vec *init_vec(size_t size, size_t add_size, mt_struct *rn_ptr, unsigned int n_orb,
                   unsigned int n_elec, dtype vec_type, int n_sites);


/*! \brief Calculate dot product
 *
 * Calculates dot product of a vector distributed across potentially many MPI
 * processes with a local sparse vector (such that the local results could be
 * added)
 *
 * \param [in] vec          Struct containing the distributed vector
 * \param [in] idx2         Indices of elements in the local vector
 * \param [in] vals2         Values of elements in the local vector
 * \param [in] num2         Number of elements in the local vector
 * \param [in] hashes2      hash values of the indices of the local vector from
 *                          the hash table of vec
 * \return the value of the dot product
 */
double vec_dot(dist_vec *vec, long long *idx2, double *vals2, size_t num2,
               unsigned long long *hashes2);


/*! \brief Generate list of occupied orbitals from bit-string representation of
 *  a determinant
 *
 * This implementation uses the procedure in Sec. 3.1 of Booth et al. (2014)
 * \param [in] det          bit string to be parsed
 * \param [in] table        byte table structure generated by gen_byte_table()
 * \param [out] occ_orbs    Occupied orbitals in the determinant
 * \return number of 1 bits in the bit string
 */
unsigned char gen_orb_list(long long det, byte_table *table, unsigned char *occ_orbs);


/*! \brief Hash function mapping vector index to processor
 *
 * \param [in] vec          Pointer to distributed sparse vector struct
 * \param [in] idx          Vector index
 * \return processor index from hash value
 */
int idx_to_proc(dist_vec *vec, long long idx);


/*! \brief Hash function mapping vector index to local hash value
 *
 * The local hash value is used to find the index on a particular processor
 *
 * \param [in] vec          Pointer to distributed sparse vector struct
 * \param [in] idx          Vector index
 * \return hash value
 */
unsigned long long idx_to_hash(dist_vec *vec, long long idx);


/*
 Add element to a buffer to be added later to the vector in an MPI step
 */
void add_int(dist_vec *vec, long long idx, int val, long long ini_flag);
void add_doub(dist_vec *vec, long long idx, double val, long long ini_flag);

/* Empty the adder by adding elements to vector, following the initiator criterion
 */
void perform_add(dist_vec *vec, long long ini_bit);


// Delete an element from the vector
void del_at_pos(dist_vec *vec, size_t pos);

// Read values from the vector
int *int_at_pos(dist_vec *vec, size_t pos);
double *doub_at_pos(dist_vec *vec, size_t pos);
unsigned char *orbs_at_pos(dist_vec *vec, size_t pos);

// Vector one-norm
double local_norm(dist_vec *vec);


/* Save/load a distributed vector to/from disk, stored in binary format with file names
 [prefix]dets[proc_rank].dat and [prefix]vals[proc_rank].dat
 */
void save_vec(dist_vec *vec, const char *path);
void load_vec(dist_vec *vec, const char *path);

#endif /* vec_utils_h */
