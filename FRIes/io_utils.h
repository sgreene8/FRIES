//
//  io_utils.h
//  FRIes
//
//  Created by Samuel Greene on 3/30/19.
//  Copyright © 2019 Samuel Greene. All rights reserved.
//

#ifndef io_utils_h
#define io_utils_h

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "csvparser.h"
#include "mpi_switch.h"
#include "fci_utils.h"
#include "det_store.h"

typedef enum {
    DOUB,
    INT
} dtype;

void read_in_doub(double *buf, char *fname);
void read_in_uchar(unsigned char *buf, char *fname);

typedef struct {
    unsigned int n_elec;
    unsigned int n_frz;
    unsigned int n_orb;
    double *eris;
    double *hcore;
    double hf_en;
    double eps;
    unsigned char *symm;
} hf_input;

typedef struct {
    unsigned int n_elec;
    unsigned int lat_len;
    unsigned int n_dim;
    double elec_int;
    double eps;
    double hf_en;
} hh_input;

/* Read in the following parameters from a Hartree-Fock calculation:
 - total number of electrons
 - number of (unfrozen) orbitals
 - number of frozen (core) electrons
 - imaginary time step (eps)
 - HF electronic energy
 - matrix of one-electron integrals
 - tensor of two-electron integrals
 
 Parameters
 ----------
 hf_dir: path to directory where the files sys_params.txt, eris.txt, hcore.txt,
    and symm.txt are stored
 in_struct: structure where the data read in will be stored
 
 Returns
 -------
 0 if successful, -1 if unsuccessful
 */
int parse_hf_input(const char *hf_dir, hf_input *in_struct);


/* Read in the following parameters for a Hubbard-Holstein calculation
 - total number of electrons
 - number of sites along one dimension of the lattice
 - number of dimensions in the lattice
 - electron-electron interaction term, U
 - imaginary time step (eps)
 - HF energy
 
 Parameters
 ----------
 hh_path: path to the file containing the Hubbard-Holstein parameters
 in_struct: structure where the data read in will be stored
 
 Returns
 -------
 0 if successful, -1 if unsuccessful
 */
int parse_hh_input(const char *hh_path, hh_input *in_struct);


/*
 Save a vector in sparse format to disk
 
 Parameters
 ----------
 path: where files should be saved
 dets: array of Slater determinant bit-string indices
 vals: array of element values in the sparse vector
 n_dets: total number of elements (including zeros) in above arrays
 el_size: size (bytes) of each element in the sparse vector
 */
void save_vec(const char *path, long long *dets, void *vals, size_t n_dets, size_t el_size);

/*
 Load a distributed sparse vector from disk
 
 Parameters
 ----------
 prefix: prefix of files containing the vector. File names should be in the
    format [prefix]dets[i].dat and [prefix]vals[i].dat, where i indicates the
    MPI process index
 dets: array in which to store Slater determinant bit-string indices
 vals: array in which to store element values
 el_size: size (bytes) of each element in the sparse vector
 
 Returns
 -------
 number of nonzero elements in the vector on this processor
 */
size_t load_vec(const char *prefix, long long *dets, void *vals, size_t el_size);


/*
 Load a single sparse vector (not distributed) in .txt format from disk
 
 Parameters
 ----------
 prefix: prefix of files containing the vector. File names should be in the
    format [prefix]dets[i].txt and [prefix]vals[i].txt, where i indicates the
    MPI process index
 dets: array in which to store Slater determinant bit-string indices
 vals: array in which to store element values
 type: data type of the vector
 
 Returns
 -------
 total number of elements in the vector
 */
size_t load_vec_txt(const char *prefix, long long *dets, void *vals, dtype type);


//size_t distribute_vec_int(long long *in_dets, int *in_vals, size_t num_in, unsigned int *proc_rns, size_t buf_len, long long (*dets_buf)[buf_len], int (*vals_buf)[buf_len], byte_table *table);

/*
 Save the random numbers from the hash function for processors (needed to reload
 vector to disk)
 
 Parameters
 ----------
 path: where file should be saved
 proc_hash: random integers used for hash function to determine the processor
 index for each determinant
 n_hash: number of random integers in the proc_hash array
 */
void save_proc_hash(const char *path, unsigned int *proc_hash, size_t n_hash);

/*
 Load the random numbers for the hash function
 used to assign determinants to processors
 
 Parameters
 ----------
 path: location of the file
 proc_hash: array in which to store random numbers for hash function
 */
void load_proc_hash(const char *path, unsigned int *proc_hash);

/*
 Calculate dot product of 2 vectors, one short enough for all of its
 elements to be enumerated, the other (long) indexed by a hash table
 for efficiency.
 
 Parameters
 ----------
 long_idx: indices of elements in long vector
 long_vals: values of elements in long vector
 short_idx: indices of elements in short vector
 short_vals: values of elements in short vector
 num_short: number of elements in short vector
 vec_hash: hash table for long vector
 short_hashes: hash values of all indices in short vector
 type: data type of long vector
 
 Returns
 -------
 dot product
 */
double calc_dprod(long long *long_idx, void *long_vals, long long *short_idx, double *short_vals, size_t num_short, hash_table *vec_hash, unsigned long long *short_hashes, dtype type);

#endif /* io_utils_h */
