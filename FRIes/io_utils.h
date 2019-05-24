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

#endif /* io_utils_h */
