/*! \file
 *
 * \brief Utilities for Hubbard-Holstein in the site basis
 */

#include "hub_holstein.h"

unsigned char gen_orb_list(long long det, byte_table *table, unsigned char *occ_orbs);


void hub_multin(long long det, unsigned int n_elec, unsigned char (*neighbors)[n_elec + 1],
                unsigned int num_sampl, mt_struct *rn_ptr, unsigned char (* chosen_orbs)[2]) {
    unsigned int samp_idx, orb_idx;
    unsigned int n_choices;
    for (samp_idx = 0; samp_idx < num_sampl; samp_idx++) {
        n_choices = neighbors[0][0] + neighbors[1][0];
        orb_idx = genrand_mt(rn_ptr) / (1. + UINT32_MAX) * n_choices;
        idx_to_orbs(orb_idx, n_elec, neighbors, chosen_orbs[samp_idx]);
    }
}


void idx_to_orbs(unsigned int chosen_idx, unsigned int n_elec,
                 unsigned char (*neighbors)[n_elec + 1], unsigned char *orbs) {
    if (chosen_idx < neighbors[0][0]) {
        orbs[0] = neighbors[0][chosen_idx + 1];
        orbs[1] = neighbors[0][chosen_idx + 1] + 1;
    }
    else if (chosen_idx < neighbors[0][0] + neighbors[1][0]){
        chosen_idx -= neighbors[0][0];
        orbs[0] = neighbors[1][chosen_idx + 1];
        orbs[1] = neighbors[1][chosen_idx + 1] - 1;
    }
    else {
        fprintf(stderr, "Error: Excitation index selected for a Hubbard determinant exceeds the possible number of excitations from that determinant.");
    }
}


size_t hub_all(long long det, unsigned int n_elec, unsigned char (*neighbors)[n_elec + 1],
               unsigned char (* chosen_orbs)[2]) {
    size_t n_ex = 0;
    unsigned int orb_idx;
    for (orb_idx = 0; orb_idx < neighbors[0][0]; orb_idx++) {
        chosen_orbs[n_ex][0] = neighbors[0][orb_idx + 1];
        chosen_orbs[n_ex][1] = neighbors[0][orb_idx + 1] + 1;
        n_ex++;
    }
    for (orb_idx = 0; orb_idx < neighbors[1][0]; orb_idx++) {
        chosen_orbs[n_ex][0] = neighbors[1][orb_idx + 1];
        chosen_orbs[n_ex][1] = neighbors[1][orb_idx + 1] - 1;
        n_ex++;
    }
    return n_ex;
}


unsigned int hub_diag(long long det, unsigned int n_sites, byte_table *table) {
    long long overlap = (det >> n_sites) & det;
    unsigned int n_overlap = 0;
    long long mask = 255;
    unsigned char det_byte;
    while (overlap != 0) {
        det_byte = overlap & mask;
        n_overlap += table->nums[det_byte];
        overlap >>= 8;
    }
    return n_overlap;
}


long long gen_neel_det_1D(unsigned int n_sites, unsigned int n_elec, unsigned int n_dim) {
    long long ones = ((1LL << n_elec) - 1) / 3;
    long long neel_state = ones << (n_sites + 1);
    
    neel_state |= ones;
    
    return neel_state;
}


double calc_ref_ovlp(long long *dets, void *vals, size_t n_dets, long long ref_det,
                     byte_table *table, dtype type) {
    size_t det_idx;
    double result = 0;
    long long curr_det;
    unsigned int n_elec = count_bits(ref_det, table);
    for (det_idx = 0; det_idx < n_dets; det_idx++) {
        curr_det = dets[det_idx];
        long long hoppers = ((curr_det & ~ref_det ) & ((curr_det & (ref_det >> 1) & (~curr_det >> 1)) | (curr_det & (ref_det << 1) & (~curr_det << 1))));
        if (count_bits(hoppers, table) == 1) {
            long long common = ref_det & curr_det & ~hoppers;
            if (count_bits(common, table) == (n_elec - 1)) {
                if (type == INT) {
                    result += ((int *)vals)[det_idx];
                }
                else if (type == DOUB) {
                    result += ((double *)vals)[det_idx];
                }
            }
        }
    }
    return result;
}
