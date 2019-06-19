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

void read_in_doub(double *buf, char *fname);
void read_in_uchar(unsigned char *buf, char *fname);

struct hf_input {
    unsigned int n_elec;
    unsigned int n_frz;
    unsigned int n_orb;
    double *eris;
    double *hcore;
    double hf_en;
    double eps;
    unsigned char *symm;
};
typedef struct hf_input hf_input;

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
int parse_input(const char *hf_dir, hf_input *in_struct);

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
 Load a sparse vector from disk
 
 Parameters
 ----------
 path: directory from which data is read
 dets: array in which to store Slater determinant bit-string indices
 vals: array in which to store element values
 el_size: size (bytes) of each element in the sparse vector
 */
size_t load_vec(const char *path, long long *dets, void *vals, size_t el_size);

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

#endif /* io_utils_h */
