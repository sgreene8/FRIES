/*
 Utilities for compressing the Hubbard-Holstein Hamiltonian matrix.
 */

#include "hub_holstein.h"

void find_neighbors(long long det, unsigned int n_sites, byte_table *table,
                    unsigned int n_elec, unsigned char (*neighbors)[n_elec + 1]) {
    long long neib_bits = det & ~(det >> 1);
    long long mask = (1LL << (2 * n_sites)) - 1;
    mask ^= (1LL << (n_sites - 1));
    mask ^= (1LL << (2 * n_sites - 1));
    neib_bits &= mask; // open boundary conditions
    neighbors[0][0] = gen_orb_list(neib_bits, table, &neighbors[0][1]);
    
    neib_bits = det & (~det << 1);
    mask = (1LL << (2 * n_sites)) - 1;
    mask ^= (1LL << n_sites);
    neib_bits &= mask; // open boundary conditions
    neighbors[1][0] = gen_orb_list(neib_bits, table, &neighbors[1][1]);
}

void hub_multin(long long det, unsigned int n_elec, unsigned char (*neighbors)[n_elec + 1],
                unsigned int num_sampl, mt_struct *rn_ptr, unsigned char (* chosen_orbs)[2]) {
    unsigned int samp_idx, orb_idx;
    unsigned int n_choices;
    for (samp_idx = 0; samp_idx < num_sampl; samp_idx++) {
        n_choices = neighbors[0][0] + neighbors[1][0];
        orb_idx = genrand_mt(rn_ptr) / MT_MAX * n_choices;
        if (orb_idx < neighbors[0][0]) {
            chosen_orbs[samp_idx][0] = neighbors[0][orb_idx + 1];
            chosen_orbs[samp_idx][1] = neighbors[0][orb_idx + 1] + 1;
        }
        else {
            orb_idx -= neighbors[0][0];
            chosen_orbs[samp_idx][0] = neighbors[1][orb_idx + 1];
            chosen_orbs[samp_idx][1] = neighbors[1][orb_idx + 1] - 1;
        }
    }
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


long long gen_hub_bitstring(unsigned int n_sites, unsigned int n_elec, unsigned int n_dim) {
    long long ones = ((1LL << n_elec) - 1) / 3;
    long long neel_state = ones << (n_sites + 1);
    
    neel_state |= ones;
    
    return neel_state;
}


int calc_ref_ovlp(long long *dets, int *vals, size_t n_dets, long long ref_det,
                  byte_table *table) {
    size_t det_idx;
    int result = 0;
    long long overlap;
    unsigned char curr_byte;
    unsigned char mask = 255;
    unsigned int n_bits;
    
    for (det_idx = 0; det_idx < n_dets; det_idx++) {
        overlap = dets[det_idx] ^ ref_det;
        n_bits = 0;
        while (overlap != 0 && n_bits == 0) {
            curr_byte = overlap & mask;
            n_bits += table->nums[curr_byte];
            overlap >>= 8;
        }
        if (n_bits == 2 && overlap == 0) {
            result += vals[det_idx];
        }
    }
    return result;
}
