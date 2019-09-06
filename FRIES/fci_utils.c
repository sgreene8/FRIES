/*! \file
 *
 * \brief Utilities for manipulating Slater determinants for FCI calculations
 */

#include "fci_utils.h"

long long gen_hf_bitstring(unsigned int n_orb, unsigned int n_elec) {
    long long ones = (1LL << (n_elec / 2)) - 1;
    long long hf_state = ones << n_orb;
    
    hf_state |= ones;
    
    return hf_state;
}


int sing_det_parity(long long *det, unsigned char *orbs) {
    *det ^= (1LL << orbs[0]);
    int sign = excite_sign(orbs[0], orbs[1], *det);
    *det ^= (1LL << orbs[1]);
    return sign;
}


int doub_det_parity(long long *det, unsigned char *orbs) {
    *det ^= (1LL << orbs[0]);
    *det ^= (1LL << orbs[1]);
    int sign = excite_sign(orbs[2], orbs[0], *det);
    sign *= excite_sign(orbs[3], orbs[1], *det);
    *det ^= (1LL << orbs[2]);
    *det ^= (1LL << orbs[3]);
    return sign;
}

int excite_sign(unsigned char cre_op, unsigned char des_op, long long det) {
    unsigned int n_perm = bits_between(det, cre_op, des_op);
    if (n_perm % 2 == 0)
        return 1;
    else
        return -1;
}
