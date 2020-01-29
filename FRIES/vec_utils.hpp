/*! \file
 *
 * \brief Utilities for storing and manipulating sparse vectors
 *
 * Supports sparse vectors distributed among multiple processes if USE_MPI is
 * defined
 */

#ifndef vec_utils_h
#define vec_utils_h

#include <cstdio>
#include <cstring>
#include <FRIES/det_store.h>
#include <FRIES/Hamiltonians/hub_holstein.hpp>
#include <FRIES/Ext_Libs/dcmt/dc.h>
#include <FRIES/mpi_switch.h>
#include <FRIES/ndarr.hpp>
#include <FRIES/compress_utils.hpp>
#include <vector>

using namespace std;

// Forward declaration from io_utils.hpp
size_t read_csv(int *buf, char *fname);
size_t read_dets(const char *path, Matrix<uint8_t> &dets);


#ifdef USE_MPI
inline void mpi_atoav(double *sendv, int *send_cts, int *disps, double *recvv, int *recv_cts) {
    MPI_Alltoallv(sendv, send_cts, disps, MPI_DOUBLE, recvv, recv_cts, disps, MPI_DOUBLE, MPI_COMM_WORLD);
}


inline void mpi_atoav(int *sendv, int *send_cts, int *disps, int *recvv, int *recv_cts) {
    MPI_Alltoallv(sendv, send_cts, disps, MPI_INT, recvv, recv_cts, disps, MPI_INT, MPI_COMM_WORLD);
}


inline void mpi_allgathv_inplace(int *arr, int *nums, int *disps) {
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, arr, nums, disps, MPI_INT, MPI_COMM_WORLD);
}


inline void mpi_allgathv_inplace(double *arr, int *nums, int *disps) {
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, arr, nums, disps, MPI_DOUBLE, MPI_COMM_WORLD);
}
#endif

template <class el_type>
class DistVec;

/*!
 * \brief Class for adding elements to a DistVec object
 * \tparam el_type Type of elements to be added to the DistVec object
 * Elements are first added to a buffer, and then the buffered elements can be distributed to the appropriate process by calling perform_add()
 */
template <class el_type>
class Adder {
public:
    /*! \brief Constructor for Adder class
     * Allocates memory for the internal buffers in the class
     * \param [in] size     Maximum number of elements per MPI process in send and receive buffers
     * \param [in] n_procs  The number of processes
     * \param [in] vec         The vector to which elements will be added
     */
    Adder(size_t size, int n_procs, DistVec<el_type> *vec) : n_bytes_(CEILING(vec->n_bits() + 1, 8)), send_idx_(n_procs, size * CEILING(vec->n_bits() + 1, 8)), send_vals_(n_procs, size), recv_idx_(n_procs, size * CEILING(vec->n_bits() + 1, 8)), recv_vals_(n_procs, size), parent_vec_(vec) {
        send_cts_ = (int *)malloc(sizeof(int) * n_procs); // 1 allocation
        recv_cts_ = (int *) malloc(sizeof(int) * n_procs);
        idx_disp_ = (int *) malloc(sizeof(int) * n_procs);
        val_disp_ = (int *) malloc(sizeof(int) * n_procs);
        for (int proc_idx = 0; proc_idx < n_procs; proc_idx++) {
            val_disp_[proc_idx] = proc_idx * (int)size;
            idx_disp_[proc_idx] = proc_idx * (int)size * n_bytes_;
            send_cts_[proc_idx] = 0;
        }
    }
    
    ~Adder() {
        free(send_cts_);
        free(recv_cts_);
        free(idx_disp_);
        free(val_disp_);
    }
    
    Adder(const Adder &a) = delete;
    
    Adder& operator= (const Adder &a) = delete;
    
    /*! \brief Remove the elements from the internal buffers and send them to the DistVec objects on their corresponding MPI processes
     */
    void perform_add();
    
    /*! \brief Add an element to the internal buffers
     * \param [in] idx      Index of the element to be added
     * \param [in] val      Value of the added element
     * \param [in] ini_flag     Either 1 or 0, indicates initiator status of the added element
     */
    void add(uint8_t *idx, el_type val, int proc_idx, int ini_flag);
private:
    Matrix<uint8_t> send_idx_; ///< Send buffer for element indices
    Matrix<el_type> send_vals_; ///< Send buffer for element values
    Matrix<uint8_t> recv_idx_; ///< Receive buffer for element indices
    Matrix<el_type> recv_vals_; ///< Receive buffer for element values
    int *send_cts_; ///< Number of elements in the send buffer for each process
    int *recv_cts_; ///< Number of elements in the receive buffer for each process
    int *idx_disp_; ///< Displacements for MPI send/receive operations for indices
    int *val_disp_; ///< Displacements for MPI send/receive operations for values
    DistVec<el_type> *parent_vec_; ///<The DistVec object to which elements are added
    uint8_t n_bytes_; ///< Number of bytes used to encode each index in the send and receive buffers
    
/*! \brief Increase the size of the buffer for temporarily storing added elements
 */
    void enlarge_() {
        printf("Increasing storage capacity in adder\n");
        size_t n_proc = send_idx_.rows();
        size_t new_idx_cols = send_idx_.cols() * 2;
        size_t new_val_cols = send_vals_.cols() * 2;
        
        int idx_counts[n_proc];
        for (int proc_idx = 0; proc_idx < n_proc; proc_idx++) {
            idx_counts[proc_idx] = send_cts_[proc_idx] * n_bytes_;
        }
        
        send_idx_.enlarge_cols(new_idx_cols, idx_counts);
        send_vals_.enlarge_cols(new_val_cols, send_cts_);
        recv_idx_.reshape(n_proc, new_idx_cols);
        recv_vals_.reshape(n_proc, new_val_cols);
        
        for (int proc_idx = 0; proc_idx < n_proc; proc_idx++) {
            idx_disp_[proc_idx] = proc_idx * (int)new_idx_cols;
            val_disp_[proc_idx] = proc_idx * (int)new_val_cols;
        }
    }
};

/*!
 * \brief Class for storing and manipulating a sparse vector
 * \tparam el_type Type of elements in the vector
 * Elements of the vector are distributed across many MPI processes, and hashing is used for efficient indexing
 */
template <class el_type>
class DistVec {
private:
    std::vector<el_type> values_; ///< Array of values of vector elements
    double *matr_el_; ///< Array of pre-calculated diagonal matrix elements associated with each vector element
    size_t n_dense_; ///< The first \p n_dense elements in the DistVec object will always be stored, even if their corresponding values are 0
    stack_entry *vec_stack_; ///< Pointer to top of stack for managing available positions in the indices array
    int n_nonz_; ///< Current number of nonzero elements in vector, including all in the dense subspace
protected:
    Matrix<uint8_t> indices_; ///< Array of indices of vector elements
    size_t max_size_; ///< Maximum number of vector elements that can be stored
    size_t curr_size_; ///< Current number of vector elements stored, including intermediate zeroes
    Matrix<uint8_t> occ_orbs_; ///< Matrix containing lists of occupied orbitals for each determniant index
    uint8_t n_bits_; ///< Number of bits used to encode each index of the vector
    byte_table *tabl_; ///< Pointer to struct used to decompose determinant indices into lists of occupied orbitals
    hash_table *vec_hash_; ///< Hash table for quickly finding indices in \p indices_
    uint64_t nonini_occ_add; ///< Number of times an addition from a noninitiator determinant to an occupied determinant occurred
private:
    Adder<el_type> adder_; ///< Pointer to adder struct for buffered addition of elements distributed across MPI processes
protected:
    
    virtual void initialize_at_pos(size_t pos) {
        values_[pos] = 0;
        matr_el_[pos] = NAN;
        uint8_t n_bytes = indices_.cols();
        if (gen_orb_list(indices_[pos], occ_orbs_[pos]) != occ_orbs_.cols()) {
            char det_txt[n_bytes * 2 + 1];
            print_str(indices_[pos], n_bytes, det_txt);
            fprintf(stderr, "Error: determinant %s created with an incorrect number of electrons.\n", det_txt);
        }
    }
    
public:
    unsigned int *proc_scrambler_; ///< Array of random numbers used in the hash function for assigning vector indices to MPI
    
    /*! \brief Constructor for DistVec object
    * \param [in] size         Maximum number of elements to be stored in the vector
    * \param [in] add_size     Maximum number of elements per processor to use in Adder object
    * \param [in] rn_ptr       Pointer to an mt_struct object for RN generation
    * \param [in] n_bits        Number of bits used to encode each index of the vector
    * \param [in] n_elec       Number of electrons represented in each vector index
     * \param [in] n_procs Number of MPI processes over which to distribute vector elements
     */
    DistVec(size_t size, size_t add_size, mt_struct *rn_ptr, uint8_t n_bits,
            unsigned int n_elec, int n_procs) : values_(size), max_size_(size), curr_size_(0), vec_stack_(NULL), occ_orbs_(size, n_elec), adder_(add_size, n_procs, this), n_nonz_(0), indices_(size, CEILING(n_bits, 8)), n_bits_(n_bits), n_dense_(0), nonini_occ_add(0) {
        matr_el_ = (double *)malloc(sizeof(double) * size);
        vec_hash_ = setup_ht(size, rn_ptr, n_bits);
        tabl_ = gen_byte_table();
    }
    
    ~DistVec() {
        free(vec_hash_);
        free(tabl_->nums);
        free(tabl_->pos);
        free(tabl_);
        free(matr_el_);
    }
    
    uint8_t n_bits() {
        return n_bits_;
    }
    
    DistVec(const DistVec &d) = delete;
    
    DistVec& operator= (const DistVec& d) = delete;

    /*! \brief Generate list of occupied orbitals from bit-string representation of
     *  a determinant
     *
     * This implementation uses the procedure in Sec. 3.1 of Booth et al. (2014)
     * \param [in] det          bit string to be parsed
     * \param [out] occ_orbs    Occupied orbitals in the determinant
     * \return number of 1 bits in the bit string
     */
    virtual uint8_t gen_orb_list(uint8_t *det, uint8_t *occ_orbs) {
        return find_bits(det, occ_orbs, indices_.cols(), tabl_);
    }

    /*! \brief Calculate dot product
     *
     * Calculates dot product of the portion of a DistVec object stored on each MPI process
     * with a local sparse vector (such that the local results could be added)
     *
     * \param [in] idx2         Indices of elements in the local vector
     * \param [in] vals2         Values of elements in the local vector
     * \param [in] num2         Number of elements in the local vector
     * \param [in] hashes2      hash values of the indices of the local vector from
     *                          the hash table of vec
     * \return the value of the dot product
     */
    double dot(Matrix<uint8_t> &idx2, double *vals2, size_t num2,
               uintmax_t *hashes2) {
        ssize_t *ht_ptr;
        double numer = 0;
        for (size_t hf_idx = 0; hf_idx < num2; hf_idx++) {
            ht_ptr = read_ht(vec_hash_, idx2[hf_idx], hashes2[hf_idx], 0);
            if (ht_ptr) {
                numer += vals2[hf_idx] * values_[*ht_ptr];
            }
        }
        return numer;
    }
    
    /*! \brief Double the maximum number of elements that can be stored */
    virtual void expand() {
        printf("Increasing storage capacity in vector\n");
        size_t new_max = max_size_ * 2;
        indices_.reshape(new_max, indices_.cols());
        matr_el_ = (double *)realloc(matr_el_, sizeof(double) * new_max);
        occ_orbs_.reshape(new_max, occ_orbs_.cols());
        values_.resize(new_max);
        max_size_ = new_max;
    }

    /*! \brief Hash function mapping vector index to MPI process
     *
     * \param [in] idx          Vector index
     * \return process index from hash value
     */
    virtual int idx_to_proc(uint8_t *idx) {
        unsigned int n_elec = (unsigned int)occ_orbs_.cols();
        uint8_t orbs[n_elec];
        gen_orb_list(idx, orbs);
        uintmax_t hash_val = hash_fxn(orbs, n_elec, NULL, 0, proc_scrambler_);
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
    virtual uintmax_t idx_to_hash(uint8_t *idx) {
        unsigned int n_elec = (unsigned int)occ_orbs_.cols();
        uint8_t orbs[n_elec];
        gen_orb_list(idx, orbs);
        return hash_fxn(orbs, n_elec, NULL, 0, vec_hash_->scrambler);
    }

    /*! \brief Add an element to the DistVec object
     *
     * The element will be added to a buffer for later processing
     *
     * \param [in] idx          The index of the element in the vector
     * \param [in] val          The value of the added element
     * \param [in] ini_flag     Either 1 or 0. If 0, will only be added to a determinant that is already occupied
     */
    void add(uint8_t *idx, el_type val, int ini_flag) {
        if (val != 0) {
            adder_.add(idx, val, idx_to_proc(idx), ini_flag);
        }
    }

    /*! \brief Incorporate elements from the Adder buffer into the vector
     *
     * Sign-coherent elements are added regardless of their corresponding initiator
     * flags. Otherwise, only elements with nonzero initiator flags are added
     */
    void perform_add() {
        adder_.perform_add();
    }
    
    /*! \brief Get the index of an unused intermediate index in the \p indices_ array, or -1 if none exists */
    ssize_t pop_stack() {
        stack_entry *head = vec_stack_;
        if (!head) {
            return -1;
        }
        ssize_t ret_idx = head->idx;
        vec_stack_ = head->next;
        free(head);
        return ret_idx;
    }
    
    /*! \brief Push an unused index in the \p indices_ array onto the stack
     * \param [in] idx The index of the available element of the \p indices array
     */
    void push_stack(size_t idx) {
        stack_entry *new_entry = (stack_entry*) malloc(sizeof(stack_entry));
        new_entry->idx = idx;
        new_entry->next = vec_stack_;
        vec_stack_ = new_entry;
    }

    /*! \brief Delete an element from the vector
     *
     * Removes an element from the vector and modifies the hash table accordingly
     *
     * \param [in] pos          The position of the element to be deleted in \p indices_
     */
    void del_at_pos(size_t pos) {
        uint8_t *idx = indices_[pos];
        uintmax_t hash_val = idx_to_hash(idx);
        push_stack(pos);
        del_ht(vec_hash_, idx, hash_val);
        n_nonz_--;
    }
    
    /*! \returns The array used to store values in the DistVec object */
    el_type *values() const {
        return (el_type *)values_.data();
    }
    
    Matrix<uint8_t> &indices() {
        return indices_;
    }
    
    /*! \returns The current number of elements in use in the \p indices_ and \p values arrays */
    size_t curr_size() const {
        return curr_size_;
    }
    
    /*! \return The maximum number of elements the vector can store*/
    size_t max_size() const {
        return max_size_;
    }
    
    /*!\returns The current number of nonzero elements in the vector */
    int n_nonz() const {
        return n_nonz_;
    }
    
    uint64_t tot_sgn_coh() const {
        int my_rank = 0;
        int n_procs = 1;
#ifdef USE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
#endif
        return sum_mpi(nonini_occ_add, my_rank, n_procs);
    }
    
    /*! \returns A pointer to the byte_table struct used to perform bit manipulations for this vector */
    byte_table *tabl() const {
        return tabl_;
    }
    
    /*! \brief Add elements destined for this process to the DistVec object
     * \param [in] indices Indices of the elements to be added
     * \param [in] vals     Values of the elements to be added
     * \param [in] count    Number of elements to be added
     */
    void add_elements(uint8_t *indices, el_type *vals, size_t count) {
        uint8_t add_n_bytes = CEILING(n_bits_ + 1, 8);
        uint8_t vec_n_bytes = indices_.cols();
        for (size_t el_idx = 0; el_idx < count; el_idx++) {
            uint8_t *new_idx = &indices[el_idx * add_n_bytes];
            int ini_flag = read_bit(new_idx, n_bits_);
            zero_bit(new_idx, n_bits_);
            uintmax_t hash_val = idx_to_hash(new_idx);
            ssize_t *idx_ptr = read_ht(vec_hash_, new_idx, hash_val, ini_flag);
            if (idx_ptr && *idx_ptr == -1) {
                *idx_ptr = pop_stack();
                if (*idx_ptr == -1) {
                    if (curr_size_ >= max_size_) {
                        expand();
                    }
                    *idx_ptr = curr_size_;
                    curr_size_++;
                }
                memcpy(indices_[*idx_ptr], new_idx, vec_n_bytes);
                initialize_at_pos(*idx_ptr);
                n_nonz_++;
            }
            int del_bool = 0;
            if (idx_ptr) {
                if (!ini_flag && values_[*idx_ptr] == 0) {
                    fprintf(stderr, "Alert: weird vector addition\n");
                }
                else {
                    nonini_occ_add += !ini_flag;
                    values_[*idx_ptr] += vals[el_idx];
                    del_bool = values_[*idx_ptr] == 0 && *idx_ptr >= n_dense_;
                }
            }
            if (del_bool == 1) {
                push_stack(*idx_ptr);
                del_ht(vec_hash_, new_idx, hash_val);
                n_nonz_--;
            }
        }
    }
    
    /*! \brief Get a pointer to a value in the \p values_ array of the DistVec object
     
    * \param [in] pos          The position of the corresponding index in the \p indices_ array
     */
    el_type *operator[](size_t pos) {
        return &values_[pos];
    }

    /*! \brief Get a pointer to the list of occupied orbitals corresponding to an
     * existing determinant index in the vector
     *
     * \param [in] pos          The row index of the index in the \p indices_  matrix
     */
    uint8_t *orbs_at_pos(size_t pos) {
        return occ_orbs_[pos];
    }
    
    /*! \brief Get a pointer to the diagonal matrix element corresponding to an element in the DistVec object
     
    * \param [in] pos          The position of the corresponding index in the \p indices_ array
     */
    double *matr_el_at_pos(size_t pos) {
        return &matr_el_[pos];
    }

    /*! \brief Calculate the sum of the magnitudes of the vector elements on each MPI process
     *
     * \return The sum of the magnitudes on each process
     */
    double local_norm() {
        double norm = 0;
        for (size_t idx = 0; idx < curr_size_; idx++) {
            norm += fabs(values_[idx]);
        }
        return norm;
    }

    /*! Save a DistVec object to disk in binary format
     *
     * The vector indices from each MPI process are stored in the file
     * [path]dets[MPI rank].dat, and the values at [path]vals[MPI rank].dat
     *
     * \param [in] path         Location where the files are to be stored
     */
    void save(const char *path)  {
        int my_rank = 0;
#ifdef USE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
#endif
        
        size_t el_size = sizeof(el_type);
        
        char buffer[300];
        sprintf(buffer, "%sdets%d.dat", path, my_rank);
        FILE *file_p = fopen(buffer, "wb");
        fwrite(indices_.data(), indices_.cols(), curr_size_, file_p);
        fclose(file_p);
        
        sprintf(buffer, "%svals%d.dat", path, my_rank);
        file_p = fopen(buffer, "wb");
        fwrite(values_.data(), el_size, curr_size_, file_p);
        fclose(file_p);
    }

    /*! Load a vector from disk in binary format
     *
     * The vector indices from each MPI process are read from the file
     * [path]dets[MPI rank].dat, and the values from [path]vals[MPI rank].dat
     *
     * \param [in] path         Location from which to read the files
     * \return Size of the dense subspace
     */
    size_t load(const char *path) {
        int my_rank = 0;
        int n_procs = 1;
#ifdef USE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
#endif
        
        char buffer[300];
        if (my_rank == 0) {
            sprintf(buffer, "%sdense.txt", path);
            int dense_sizes[n_procs];
            read_csv(dense_sizes, buffer);
#ifdef USE_MPI
            MPI_Scatter(dense_sizes, 1, MPI_INT, &n_dense_, 1, MPI_INT, 0, MPI_COMM_WORLD);
#else
            n_dense_ = dense_sizes[my_rank];
#endif
        }
        
        size_t el_size = sizeof(el_type);
        
        size_t n_dets;
        size_t n_bytes = indices_.cols();
        sprintf(buffer, "%sdets%d.dat", path, my_rank);
        FILE *file_p = fopen(buffer, "rb");
        if (!file_p) {
            fprintf(stderr, "Error: could not open saved binary vector file at %s\n", buffer);
            return n_dense_;
        }
        n_dets = fread(indices_.data(), n_bytes, 10000000, file_p);
        fclose(file_p);
        
        sprintf(buffer, "%svals%d.dat", path, my_rank);
        file_p = fopen(buffer, "rb");
        if (!file_p) {
            fprintf(stderr, "Error: could not open saved binary vector file at %s\n", buffer);
            return n_dense_;
        }
        fread(values_.data(), el_size, n_dets, file_p);
        fclose(file_p);
        
        n_nonz_ = 0;
        for (size_t det_idx = 0; det_idx < n_dets; det_idx++) {
            int is_nonz = 0;
            double value = 0;
            if (fabs(values_[det_idx]) > 1e-9 || det_idx < n_dense_) {
                is_nonz = 1;
                value = values_[det_idx];
            }
            if (is_nonz) {
                uint8_t *new_idx = indices_[det_idx];
                uintmax_t hash_val = idx_to_hash(new_idx);
                ssize_t *idx_ptr = read_ht(vec_hash_, new_idx, hash_val, 1);
                *idx_ptr = n_nonz_;
                memmove(indices_[n_nonz_], new_idx, n_bytes);
                initialize_at_pos(n_nonz_);
                values_[n_nonz_] = value;
                n_nonz_++;
            }
        }
        curr_size_ = n_nonz_;
        return n_dense_;
    }
    
    /*! \brief Load all of the vector indices defining the dense subspace from disk, and initialize the corresponding vector elements to 0
     *
     * Indices must be stored on disk as ≤64-bit integers
     *
     * \param [in] read_path     Path to the file where the indices are stored
     * \param [in] save_dir      Directory in which to store a file containing the length of the dense subspace on each MPI process
     * \return Size of the dense subspace
     */
    size_t init_dense(const char *read_path, const char *save_dir) {
        size_t n_loaded = read_dets(read_path, indices_);
        for (size_t idx = 0; idx < n_loaded; idx++) {
            add(indices_[idx], 1, 1);
        }
        perform_add();
        
        n_dense_ = curr_size_;
        bzero(values_.data(), n_dense_ * sizeof(el_type));

        int n_procs = 1;
        int my_rank = 0;
#ifdef USE_MPI
        MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
#endif
        int dense_sizes[n_procs];
        dense_sizes[my_rank] = (int) n_dense_;
#ifdef USE_MPI
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_INT, dense_sizes, 1, MPI_INT, MPI_COMM_WORLD);
#endif
        if (my_rank == 0) {
            char buf[200];
            sprintf(buf, "%sdense.txt", save_dir);
            FILE *dense_f = fopen(buf, "w");
            if (!dense_f) {
                fprintf(stderr, "Error opening file at %s\n", buf);
                return n_dense_;
            }
            for (int proc_idx = 0; proc_idx < n_procs; proc_idx++) {
                fprintf(dense_f, "%d,", dense_sizes[proc_idx]);
            }
            fprintf(dense_f, "\n");
            fclose(dense_f);
        }
        return n_dense_;
    }
    
    
    /*! \brief Calculate the one-norm of the vector in the dense subspace
     * \return The total norm from all processes
     */
    el_type dense_norm() {
        el_type result = 0;
        el_type element;
        for (size_t det_idx = 0; det_idx < n_dense_; det_idx++) {
            element = values_[det_idx];
            result += element >= 0 ? element : -element;
        }
        int n_procs = 1;
        int my_rank = 0;
#ifdef USE_MPI
        MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
#endif
        el_type glob_norm;
        glob_norm = sum_mpi(result, my_rank, n_procs);
        return glob_norm;
    }
    
    /*! \brief Collect all of the vector elements from other MPI processes and accumulate them in the vector on each process */
    void collect_procs() {
        int n_procs = 1;
        int my_rank = 0;
#ifdef USE_MPI
        MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
#endif
        int vec_sizes[n_procs];
        int idx_sizes[n_procs];
        int n_bytes = (int) indices_.cols();
        vec_sizes[my_rank] = (int)curr_size_;
        idx_sizes[my_rank] = (int)curr_size_ * n_bytes;
#ifdef USE_MPI
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_INT, vec_sizes, 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_INT, idx_sizes, 1, MPI_INT, MPI_COMM_WORLD);
#endif
        int tot_size = 0;
        int disps[n_procs];
        int idx_disps[n_procs];
        for (int proc_idx = 0; proc_idx < n_procs; proc_idx++) {
            idx_disps[proc_idx] = tot_size * n_bytes;
            disps[proc_idx] = tot_size;
            tot_size += vec_sizes[proc_idx];
        }
        size_t el_size = sizeof(el_type);
        if (tot_size > max_size_) {
            indices_.reshape(tot_size, n_bytes);
            values_.resize(tot_size);
        }
        memmove(indices_[disps[my_rank]], indices_.data(), vec_sizes[my_rank] * n_bytes);
        memmove(&values_.data()[disps[my_rank]], values_.data(), vec_sizes[my_rank] * el_size);
#ifdef USE_MPI
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, indices_.data(), idx_sizes, idx_disps, MPI_UINT8_T, MPI_COMM_WORLD);
        mpi_allgathv_inplace(values_.data(), vec_sizes, disps);
#endif
        curr_size_ = tot_size;
    }
};


template <class el_type>
void Adder<el_type>::add(uint8_t *idx, el_type val, int proc_idx, int ini_flag) {
    int *count = &send_cts_[proc_idx];
    if (*count == send_vals_.cols()) {
        enlarge_();
    }
    uint8_t *cpy_idx = &send_idx_[proc_idx][*count * n_bytes_];
    uint8_t idx_bits = parent_vec_->n_bits();
    cpy_idx[n_bytes_ - 1] = 0; // To prevent buffer overflow in hash function after elements are added
    memcpy(cpy_idx, idx, CEILING(idx_bits, 8));
    if (ini_flag) {
        set_bit(cpy_idx, idx_bits);
    }
    send_vals_(proc_idx, *count) = val;
    (*count)++;
}

template <class el_type>
void Adder<el_type>::perform_add() {
    int n_procs = 1;
    
    size_t el_size = sizeof(el_type);
#ifdef USE_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Alltoall(send_cts_, 1, MPI_INT, recv_cts_, 1, MPI_INT, MPI_COMM_WORLD);
    
    int send_idx_cts[n_procs];
    int recv_idx_cts[n_procs];
    for (int proc_idx = 0; proc_idx < n_procs; proc_idx++) {
        send_idx_cts[proc_idx] = send_cts_[proc_idx] * n_bytes_;
        recv_idx_cts[proc_idx] = recv_cts_[proc_idx] * n_bytes_;
    }
    
    MPI_Alltoallv(send_idx_.data(), send_idx_cts, idx_disp_, MPI_UINT8_T, recv_idx_.data(), recv_idx_cts, idx_disp_, MPI_UINT8_T, MPI_COMM_WORLD);
    mpi_atoav(send_vals_.data(), send_cts_, val_disp_, recv_vals_.data(), recv_cts_);
#else
    for (int proc_idx = 0; proc_idx < n_procs; proc_idx++) {
        int cpy_size = send_cts_[proc_idx];
        recv_cts_[proc_idx] = cpy_size;
        memcpy(recv_idx_[proc_idx], send_idx_[proc_idx], cpy_size * n_bytes_);
        memcpy(recv_vals_[proc_idx], send_vals_[proc_idx], cpy_size * el_size);
    }
#endif
    // Move elements from receiving buffers to vector
    for (int proc_idx = 0; proc_idx < n_procs; proc_idx++) {
        send_cts_[proc_idx] = 0;
        parent_vec_->add_elements(recv_idx_[proc_idx], recv_vals_[proc_idx], recv_cts_[proc_idx]);
    }
}


#endif /* vec_utils_h */
