/*! \file
 *
 * \brief Utilities for Hubbard-Holstein in the site basis
 */

#include "hub_holstein.hpp"


void hub_multin(unsigned int n_elec, const uint8_t *neighbors,
                unsigned int num_sampl, mt_struct *rn_ptr, uint8_t (* chosen_orbs)[2]) {
    unsigned int samp_idx, orb_idx;
    unsigned int n_choices;
    for (samp_idx = 0; samp_idx < num_sampl; samp_idx++) {
        n_choices = neighbors[0] + neighbors[n_elec + 1];
        orb_idx = genrand_mt(rn_ptr) / (1. + UINT32_MAX) * n_choices;
        idx_to_orbs(orb_idx, n_elec, neighbors, chosen_orbs[samp_idx]);
    }
}


void idx_to_orbs(unsigned int chosen_idx, unsigned int n_elec,
                 const uint8_t *neighbors, uint8_t *orbs) {
    if (chosen_idx < neighbors[0]) {
        orbs[0] = neighbors[chosen_idx + 1];
        orbs[1] = neighbors[chosen_idx + 1] + 1;
    }
    else if (chosen_idx < neighbors[0] + neighbors[n_elec + 1]){
        chosen_idx -= neighbors[0];
        orbs[0] = neighbors[n_elec + 1 + chosen_idx + 1];
        orbs[1] = neighbors[n_elec + 1 + chosen_idx + 1] - 1;
    }
    else {
        fprintf(stderr, "Error: Excitation index selected for a Hubbard determinant exceeds the possible number of excitations from that determinant.");
    }
}


uint8_t idx_of_doub(unsigned int chosen_idx, unsigned int n_elec,
                    const uint8_t *occ, const uint8_t *det, unsigned int n_sites) {
    uint8_t n_doub = 0;
    for (size_t elec_idx = 0; elec_idx < n_elec / 2; elec_idx++) {
        if (read_bit(det, occ[elec_idx] + n_sites)) {
            if (n_doub == chosen_idx) {
                return occ[elec_idx];
            }
            n_doub++;
        }
    }
    fprintf(stderr, "Error in idx_of_doub: index %u not found\n", chosen_idx);
    return 255;
}


uint8_t idx_of_sing(unsigned int chosen_idx, unsigned int n_elec,
                    const uint8_t *occ, const uint8_t *det, unsigned int n_sites) {
    uint8_t n_sing = 0;
    for (size_t elec_idx = 0; elec_idx < n_elec / 2; elec_idx++) {
        if (read_bit(det, occ[elec_idx] + n_sites) == 0) {
            if (n_sing == chosen_idx) {
                return occ[elec_idx];
            }
            n_sing++;
        }
    }
    for (size_t elec_idx = n_elec / 2; elec_idx < n_elec; elec_idx++) {
        if (read_bit(det, occ[elec_idx] - n_sites) == 0) {
            if (n_sing == chosen_idx) {
                return occ[elec_idx];
            }
            n_sing++;
        }
    }
    fprintf(stderr, "Error in idx_of_doub: index %u not found\n", chosen_idx);
    return 255;
}


size_t hub_all(unsigned int n_elec, uint8_t *neighbors,
               uint8_t (* chosen_orbs)[2]) {
    size_t n_ex = 0;
    unsigned int orb_idx;
    for (orb_idx = 0; orb_idx < neighbors[0]; orb_idx++) {
        chosen_orbs[n_ex][0] = neighbors[orb_idx + 1];
        chosen_orbs[n_ex][1] = neighbors[orb_idx + 1] + 1;
        n_ex++;
    }
    for (orb_idx = 0; orb_idx < neighbors[n_elec + 1]; orb_idx++) {
        chosen_orbs[n_ex][0] = neighbors[n_elec + 1 + orb_idx + 1];
        chosen_orbs[n_ex][1] = neighbors[n_elec + 1 + orb_idx + 1] - 1;
        n_ex++;
    }
    return n_ex;
}


unsigned int hub_diag(uint8_t *det, unsigned int n_sites, byte_table *table) {
    unsigned int n_overlap = 0;
    size_t byte_idx;
    
    uint8_t later_byte;
    uint8_t mask;
    
    // take care of all the full bytes
    for (byte_idx = 0; byte_idx < n_sites / 8; byte_idx++) {
        later_byte = det[n_sites / 8 + byte_idx] >> n_sites % 8;
        mask = later_byte & det[byte_idx];
        n_overlap += table->nums[mask];

        later_byte = det[n_sites / 8 + byte_idx + 1] << (8 - (n_sites % 8));
        mask = later_byte & det[byte_idx];
        n_overlap += table->nums[mask];
    }
    if (n_sites % 8) {
        later_byte = det[n_sites / 8 + byte_idx];
        if ((1 + byte_idx + n_sites / 8) * 8 > (2 * n_sites)) {
            later_byte &= (1 << (2 * n_sites % 8)) - 1;
        }
        later_byte >>= n_sites % 8;
        mask = later_byte & det[byte_idx];
        n_overlap += table->nums[mask];
    }
    
    if ((n_sites / 8 + byte_idx + 1) < CEILING(2 * n_sites, 8)) {
        later_byte = det[n_sites / 8 + byte_idx + 1];
        later_byte &= (1 << (2 * n_sites % 8)) - 1;
        later_byte <<= (8 - (n_sites % 8));
        mask = later_byte & det[byte_idx];
        n_overlap += table->nums[mask];
    }
    return n_overlap;
}


void gen_neel_det_1D(unsigned int n_sites, unsigned int n_elec, uint8_t ph_bits,
                     uint8_t *det) {
    size_t byte_idx;
    for (byte_idx = 0; byte_idx < n_elec / 8; byte_idx++) {
        det[byte_idx] = 255 / 3;
    }
    det[byte_idx] = ((1 << (n_elec % 8)) - 1) / 3;
    
    for (byte_idx++; byte_idx < CEILING(n_sites, 8); byte_idx++) {
        det[byte_idx] = 0;
    }
    
    byte_idx = n_sites / 8;
    det[byte_idx] |= (255 / 3) << ((n_sites + 1) % 8);
    
    for (byte_idx++; byte_idx < (n_sites + n_elec) / 8; byte_idx++) {
        det[byte_idx] = (255 / 3) << 1;
    }
    byte_idx = (n_sites + n_elec) / 8;
    if (byte_idx == n_sites / 8) {
        det[byte_idx] &= (1 << (n_sites % 8)) - 1;
        det[byte_idx] |= ((1 << (n_elec % 8)) / 3) << (1 + n_sites % 8);
    }
    else if ((n_sites + n_elec) % 8 != 0){
        det[byte_idx] = ((1 << ((n_sites + n_elec) % 8)) / 3) << 1;
    }
    for (byte_idx++; byte_idx < CEILING((2 + ph_bits) * n_sites, 8); byte_idx++) {
        det[byte_idx] = 0;
    }
}

