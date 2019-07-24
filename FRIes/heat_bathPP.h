/*
 Utilities for compressing the Heat-Bath Power-Pitzer matrix
 */

#ifndef heat_bathPP_h
#define heat_bathPP_h

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    size_t n_orb;
    // single-electron elements of HB-PP matrix, length n_orb
    double *s_tens;
    // double-electron elements of HB-PP matrix for same spins, length
    // n_orb choose 2
    double *d_same;
    // double-electron elements of HB-PP matrix for same spins, dimensions
    // n_orb x n_orb
    double *d_diff;
} hb_info;

/*
 Calculate the D matrix and S vector used to calculate elements of the HB-PP
 matrix for occupied orbitals in double excitations.
 
 Parameters
 ----------
 n_frozen: total number of electrons frozen in the calculation
 n_orb: dimension of the eris 2-electron tensor
 eris: 2-electron integrals, of dimensions tot_orb x tot_orb x tot_orb x tot_orb
 
 Returns
 -------
 hb_info object containing the single- and double-electron probabilities
 */
hb_info *set_up(unsigned int tot_orb, unsigned int n_orb,
                double (*eris)[tot_orb][tot_orb][tot_orb]);

#endif /* heat_bathPP_h */
