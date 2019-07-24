//
//  fci_utils.h
//  FRIes
//
//  Created by Samuel Greene on 3/31/19.
//  Copyright © 2019 Samuel Greene. All rights reserved.
//

#ifndef fci_utils_h
#define fci_utils_h

#include <stdio.h>
#include <stdlib.h>

/*Generate the bit string representation of the Hartree-Fock determinant.
 
 Arguments
 ----------
 n_orb: the number of spatial orbitals in the basis
 n_elec: the number of electrons in the system
 
 Returns: bit string representation of the determinant
 */
long long gen_hf_bitstring(unsigned int n_orb, unsigned int n_elec);

/* Generate a list of all spatial orbitals of each irrep in the point
 group.
 
 Arguments
 ---------
 orb_symm: 1-D array with irreps of each orbital
 n_orbs: number of HF spatial orbitals
 n_symm: number of distinct irreps in this group
 lookup_tabl: (n_symm x (n_orbs + 1)) 2-D array. Upon return, 0th column lists
    number of orbitals with each irrep, and subsequent columns list those orbitals.
 */
void gen_symm_lookup(unsigned char *orb_symm, unsigned int n_orbs, unsigned int n_symm,
                     unsigned char (* lookup_tabl)[n_orbs + 1]);

/* Calculate the matrix element for a double excitation without accounting
 for the parity of the excitation.
 
 Arguments
 ---------
 chosen_idx : the chosen indices of the two occupied orbitals (0th and 1st
    elements) and two virtual orbitals (2nd and 3rd elements) in the
    excitation
 n_orbs: number of HF spatial orbitals (including frozen)
 eris: 4-D array of 2-electron integrals in spatial MO basis
 n_frozen: number of core electrons frozen in the calculation
 
 Returns
 -------
 calculated matrix element
 */
double doub_matr_el_nosgn(unsigned char *chosen_idx, unsigned int n_orbs,
                          double (* eris)[n_orbs][n_orbs][n_orbs], unsigned int n_frozen);

/* Calculate the bit-string representation of a new determinant resulting from
 an excitation from a specified determinant, and the sign of the excitation.
 
 Arguments
 ---------
 det: pointer to origin determinant. upon return, points to new determinant
 orbs: the chosen indices of the two occupied orbitals (0th and 1st
    elements) and two virtual orbitals (2nd and 3rd elements) in the
    excitation
 
 Returns
 -------
 sign (+1 or -1) of the excitation
 */
int doub_det_parity(long long *det, unsigned char *orbs);


/* Calculate the matrix element for a single excitation without accounting
 for the parity of the excitation.
 
 Arguments
 ---------
 chosen_idx: the chosen indices of the occupied orbital (0th element) and
    virtual orbital (1st element) in the excitation
 occ_orbs: orbitals occupied in the origin determinant
 n_orbs: number of HF spatial orbitals (including frozen)
 eris: 4-D array of 2-electron integrals in spatial MO basis
 h_core: 2-D array of 1-electron integrals in spatial MO basis
 n_frozen: number of core electrons frozen in the calculation
 n_elec: number of unfrozen electrons in the system
 
 Returns
 -------
 calculated matrix element
 */
double sing_matr_el_nosgn(unsigned char *chosen_idx, unsigned char *occ_orbs,
                          unsigned int n_orbs, double (* eris)[n_orbs][n_orbs][n_orbs],
                          double (* h_core)[n_orbs], unsigned int n_frozen,
                          unsigned int n_elec);


/* Calculate the bit-string representation of a new determinant resulting from
 an excitation from a specified determinant, and the sign of the excitation.
 
 Arguments
 ---------
 det: pointer to origin determinant. upon return, points to new determinant
 orbs:  the chosen indices of the occupied orbital (0th element) and
    virtual orbital (1st element) in the excitation
 
 Returns
 -------
 sign (+1 or -1) of the excitation
 */
int sing_det_parity(long long *det, unsigned char *orbs);


/* Calculate the parity of a single excitation for a bit string
 representation of a determinants, i.e. determine the sign of cre_op^+
 des_op |det>. Same as the pyscf subroutine pyscf.fci.cistring.cre_des_sign
 
 Returns
 -------
 sign (+1 or -1) of the excitation
 */
int excite_sign(unsigned char cre_op, unsigned char des_op, long long det);

/* count number of 1's between bits a and b in binary representation of bit_str.
 order of a and b does not matter
 
 Returns
 -------
 number of bits
 */
unsigned int bits_between(long long bit_str, unsigned char a, unsigned char b);

/* Generate all spin- and symmetry allowed double excitations from a Slater determinant.
 
 Arguments
 ---------
 det: bit string representation of determinant
 occ_orbs: array of length (num_elec) listing occupied orbitals in det
 num_elec: Number of electrons represented in det
 num_orb: Total number of spatial HF orbitals in the basis
 res_arr: (n x 4) 2-d array containing the occupied (0th and 1st) and unoccupied
    (2nd and 3rd) orbitals involved in each excitation
 symm: irrep of each of the orbitals in the HF basis
 
 Returns
 -------
 total number of excitations generated
 */
size_t _doub_ex_symm(long long det, unsigned char *occ_orbs, unsigned int num_elec,
                     unsigned int num_orb, unsigned char res_arr[][4], unsigned char *symm);

/* Generate all symmetry-allowed double excitations from the Hartree-Fock determinant.
 
 Arguments
 ---------
 hf_det: bit-string representation of the HF determinant
 hf_occ: orbitals occupied in the HF determinant
 num_elec: number of electrons in the system
 n_orb: number of spatial orbitals in the basis (including frozen)
 orb_symm: symmetry of each spatial orbital in the basis
 eris: 4-D array of 2-electron integrals in spatial MO basis
 n_frozen: number of core electrons frozen in the calculation
 ex_dets: bit-string representation of each determinant resulting from a double excitation
 ex_mel: matrix element for each excitation
 
 Returns
 -------
 total number of excitations generated
 */
size_t gen_hf_ex(long long hf_det, unsigned char *hf_occ, unsigned int num_elec,
                 unsigned int n_orb, unsigned char *orb_symm, double (*eris)[n_orb][n_orb][n_orb],
                 unsigned int n_frozen, long long *ex_dets, double *ex_mel);

/* Calculate number of double excitations from any determinant without
 accounting for point-group symmetry
 
 Arguments
 ---------
 num_elec: number of electrons in the system
 n_orb: number of spatial orbitals in the basis
 
 Returns
 -------
 number of double excitations
 */
size_t count_doub_nosymm(unsigned int num_elec, unsigned int num_orb);

/* Count the number of spin- and symmetry-allowed single excitations from a
 given determinant.
 
 Arguments
 ---------
 det: bit-string representation of the origin determinant
 occ_orbs: list of occupied orbitals in det
 orb_symm: symmetry of each spatial orbital in the basis
 num_orb: number of unfrozen orbitals in the HF basis
 lookup_tabl: List of orbitals with each type of symmetry, as generated by
    gen_symm_lookup()
 num_elec: number of electrons represented in det
 
 Returns
 -------
 number of symmetry-allowed single excitations
 */
size_t count_singex(long long det, unsigned char *occ_orbs, unsigned char *orb_symm,
                    unsigned int num_orb, unsigned char (* lookup_tabl)[num_orb + 1],
                    unsigned int num_elec);


/* Calculate diagonal Hamiltonian matrix element for a Slater determinant
 
 Arguments
 ---------
 occ_orbs: orbitals occupied in the determinant
 n_orbs: number of HF spatial orbitals (including frozen)
 eris: 4-D array of 2-electron integrals in spatial MO basis
 h_core: 2-D array of 1-electron integrals in spatial MO basis
 n_frozen: number of core electrons frozen in the calculation
 n_elec: number of unfrozen electrons in the system
 
 Returns
 -------
 calculated matrix element
 */
double diag_matrel(unsigned char *occ_orbs, unsigned int n_orbs,
                   double (* eris)[n_orbs][n_orbs][n_orbs], double (* h_core)[n_orbs],
                   unsigned int n_frozen, unsigned int n_elec);

#endif /* fci_utils_h */
