/*! \file
 *
 * \brief Utilities for manipulating Slater determinants for FCI calculations
 */

#include "fci_utils.h"

void gen_hf_bitstring(unsigned int n_orb, unsigned int n_elec, uint8_t *det) {
    uint8_t byte_idx;
    for (byte_idx = 0; byte_idx < (n_elec / 2 / 8); byte_idx++) {
        det[byte_idx] = 255;
    }
    det[n_elec / 2 / 8] = (1 << (n_elec / 2 % 8)) - 1;
    
    for (byte_idx = n_elec / 2 / 8 + 1; byte_idx <= n_orb / 8; byte_idx++) {
        det[byte_idx] = 0;
    }
    
    byte_idx = n_orb / 8;
    det[byte_idx] |= ~((1 << (n_orb % 8)) - 1);
    byte_idx++;
    for (; byte_idx < ((n_orb + n_elec / 2) / 8); byte_idx++) {
        det[byte_idx] = 255;
    }
    byte_idx = (n_orb + n_elec / 2) / 8;
    det[byte_idx] = (1 << ((n_orb + n_elec / 2) % 8)) - (1 << ((n_orb / 8 == byte_idx) ? (n_orb % 8) : 0));
    
    for (byte_idx++; byte_idx < CEILING(2 * n_orb, 8); byte_idx++) {
        det[byte_idx] = 0;
    }
}


int sing_det_parity(uint8_t *det, uint8_t *orbs) {
    zero_bit(det, orbs[0]);
    int sign = excite_sign(orbs[0], orbs[1], det);
    set_bit(det, orbs[1]);
    return sign;
}


int doub_det_parity(uint8_t *det, uint8_t *orbs) {
    zero_bit(det, orbs[0]);
    zero_bit(det, orbs[1]);
    int sign = excite_sign(orbs[2], orbs[0], det);
    sign *= excite_sign(orbs[3], orbs[1], det);
    set_bit(det, orbs[2]);
    set_bit(det, orbs[3]);
    return sign;
}

int excite_sign(uint8_t cre_op, uint8_t des_op, uint8_t *det) {
    unsigned int n_perm = bits_between(det, cre_op, des_op);
    if (n_perm % 2 == 0)
        return 1;
    else
        return -1;
}

uint8_t find_nth_virt(uint8_t *occ_orbs, int spin, uint8_t n_elec,
                      uint8_t n_orb, uint8_t n) {
    uint8_t virt_orb = n_orb * spin + n;
    for (size_t orb_idx = n_elec / 2 * spin; occ_orbs[orb_idx] <= virt_orb; orb_idx++) {
        if (occ_orbs[orb_idx] <= virt_orb) {
            virt_orb++;
        }
    }
    return virt_orb;
}
