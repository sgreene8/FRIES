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
#include <FRIES/det_store.h>
#include <FRIES/Hamiltonians/hub_holstein.h>
#include <FRIES/Ext_Libs/dcmt/dc.h>
#include <FRIES/mpi_switch.h>


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


/*! \brief Add an int to a vector
 *
 * The element will be added to a buffer for later processing
 *
 * \param [in] vec          The dist_vec struct to which the element will be
 *                          added
 * \param [in] idx          The index of the element in the vector
 * \param [in] val          The value of the added element
 * \param [in] ini_flag     A bit string indicating whether the added element
 *                          came from an initiator element. None of the 1 bits
 *                          should overlap with the orbitals encoded in the rest
 *                          of the bit string
 */
void add_int(dist_vec *vec, long long idx, int val, long long ini_flag);


/*! \brief Add a double to a vector
 *
 * The element will be added to a buffer for later processing
 *
 * \param [in] vec          The dist_vec struct to which the element will be
 *                          added
 * \param [in] idx          The index of the element in the vector
 * \param [in] val          The value of the added element
 * \param [in] ini_flag     A bit string indicating whether the added element
 *                          came from an initiator element. None of the 1 bits
 *                          should overlap with the orbitals encoded in the rest
 *                          of the bit string
 */
void add_doub(dist_vec *vec, long long idx, double val, long long ini_flag);


/*! \brief Incorporate elements from the buffer into the vector
 *
 * Sign-coherent elements are added regardless of their corresponding initiator
 * flags. Otherwise, only elements with nonzero initiator flags are added
 *
 * \param [in] vec          The dist_vec struct on which to perform addition
 * \param [in] ini_bit      A bit mask defining where to look for initiator
 *                          flags in added elements
 */
void perform_add(dist_vec *vec, long long ini_bit);


/*! \brief Delete an element from the vector
 *
 * Removes an element from the vector and modifies the hash table accordingly
 *
 * \param [in] vec          The dist_vec struct from which to delete the element
 * \param [in] pos          The position of the element to be deleted in the
 *                          element storage array of the dist_vec structure
 */
void del_at_pos(dist_vec *vec, size_t pos);

/*! \brief Get a pointer to an element in the vector
 *
 * This function must be used in lieu of \p vec->values[\p pos] because the
 * values are stored as a (void *) array
 *
 * \param [in] vec          The dist_vec structure from which to read the element
 * \param [in] pos          The position of the desired element in the local
 *                          storage
 */
int *int_at_pos(dist_vec *vec, size_t pos);


/*! \brief Get a pointer to an element in the vector
 *
 * This function must be used in lieu of \p vec->values[\p pos] because the
 * values are stored as a (void *) array
 *
 * \param [in] vec          The dist_vec structure from which to read the element
 * \param [in] pos          The position of the desired element in the local
 *                          storage
 */
double *doub_at_pos(dist_vec *vec, size_t pos);


/*! \brief Get a pointer to the list of occupied orbitals corresponding to an
 * existing determinant index in the vector
 *
 * \param [in] vec          The dist_vec structure to reference
 * \param [in] pos          The position of the index in the local storage
 */
unsigned char *orbs_at_pos(dist_vec *vec, size_t pos);


/*! \brief Calculate the one-norm of a vector
 *
 * This function sums over the elements on all MPI processes
 *
 * \param [in] vec          The vector whose one-norm is to be calculated
 * \return The one-norm of the vector
 */
double local_norm(dist_vec *vec);


/*! Save a vector to disk in binary format
 *
 * The vector indices from each MPI process are stored in the file
 * [path]dets[MPI rank].dat, and the values at [path]vals[MPI rank].dat
 *
 * \param [in] vec          Pointer to the vector to save
 * \param [in] path         Location where the files are to be stored
 */
void save_vec(dist_vec *vec, const char *path);


/*! Load a vector from disk in binary format
 *
 * The vector indices from each MPI process are read from the file
 * [path]dets[MPI rank].dat, and the values from [path]vals[MPI rank].dat
 *
 * \param [out] vec         Pointer to an allocated and initialized dist_vec
 *                          struct
 * \param [in] path         Location where the files are to be stored
 */
void load_vec(dist_vec *vec, const char *path);

#endif /* vec_utils_h */
