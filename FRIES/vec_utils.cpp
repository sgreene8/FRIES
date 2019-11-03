/*! \file
 *
 * \brief Utilities for storing and manipulating sparse vectors
 *
 * Supports sparse vectors distributed among multiple processes if USE_MPI is
 * defined
 */

#include "vec_utils.hpp"


unsigned char gen_orb_list(long long det, byte_table *table, unsigned char *occ_orbs) {
    unsigned int byte_idx, elec_idx;
    long long mask = 255;
    unsigned char n_elec, det_byte, bit_idx;
    elec_idx = 0;
    byte_idx = 0;
    unsigned char tot_elec = 0;
    while (det != 0) {
        det_byte = det & mask;
        n_elec = table->nums[det_byte];
        for (bit_idx = 0; bit_idx < n_elec; bit_idx++) {
            occ_orbs[elec_idx + bit_idx] = (8 * byte_idx + table->pos[det_byte][bit_idx]);
        }
        elec_idx = elec_idx + n_elec;
        det = det >> 8;
        byte_idx = byte_idx + 1;
        tot_elec += n_elec;
    }
    return tot_elec;
}


void find_neighbors_1D(long long det, unsigned int n_sites, byte_table *table,
                       unsigned int n_elec, unsigned char *neighbors) {
    long long neib_bits = det & ~(det >> 1);
    long long mask = (1LL << (2 * n_sites)) - 1;
    mask ^= (1LL << (n_sites - 1));
    mask ^= (1LL << (2 * n_sites - 1));
    neib_bits &= mask; // open boundary conditions
    neighbors[0] = gen_orb_list(neib_bits, table, &neighbors[1]);
    
    neib_bits = det & (~det << 1);
    mask = (1LL << (2 * n_sites)) - 1;
    mask ^= (1LL << n_sites);
    neib_bits &= mask; // open boundary conditions
    neighbors[n_elec + 1] = gen_orb_list(neib_bits, table, &neighbors[n_elec + 1 + 1]);
}
