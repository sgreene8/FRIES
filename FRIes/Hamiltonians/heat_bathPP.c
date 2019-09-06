/*! \file
 *
 * \brief Utilities for Heat-Bath Power-Pitzer compression of Hamiltonian
 *
 * These functions apply only to the double excitation portion of a molecular
 * Hamiltonian matrix. Single excitations are treated as in the near-uniform
 * scheme
 */

#include "heat_bathPP.h"

#define TRI_N(n)((n) * (n + 1) / 2)
#define I_J_TO_TRI(i, j)(TRI_N(j - 1) + i)

hb_info *set_up(unsigned int tot_orb, unsigned int n_orb,
                double (*eris)[tot_orb][tot_orb][tot_orb]) {
    hb_info *hb_obj = malloc(sizeof(hb_info));
    hb_obj->n_orb = n_orb;
    unsigned int half_frz = tot_orb - n_orb;
    
    double (*d_diff)[n_orb] = calloc(n_orb * n_orb, sizeof(double));
    size_t i, j, a, b;
    for (i = 0; i < n_orb; i++) {
        for (j = 0; j < n_orb; j++) {
            for (a = half_frz; a < tot_orb; a++) {
                for (b = half_frz; b < tot_orb; b++) {
                    if (i != (a - half_frz) && j != (b - half_frz)) {
                        d_diff[i][j] += fabs(eris[i + half_frz][j + half_frz][a][b]); // exchange terms are zero
                    }
                }
            }
        }
    }
    hb_obj->d_diff = (double *)d_diff;
    
    double *d_same = calloc(n_orb * (n_orb - 1) / 2, sizeof(double));
    size_t tri_idx = 0;
    for (j = 1; j < n_orb; j++) {
        for (i = 0; i < j; i++) {
            for (a = half_frz; a < tot_orb; a++) {
                for (b = half_frz; b < a; b++) {
                    if ((a - half_frz) != j && (a - half_frz) != i && (b - half_frz) != j && (b - half_frz) != i) {
                        d_same[tri_idx] += 2 * fabs(eris[i + half_frz][j + half_frz][a][b] - eris[i + half_frz][j + half_frz][b][a]);
                    }
                }
            }
            tri_idx++;
        }
    }
    hb_obj->d_same = d_same;
    
    double *s_tens = calloc(n_orb, sizeof(double));
    for (i = 0; i < n_orb; i++) {
        for (j = 0; j < i; j++) {
            s_tens[i] += d_same[I_J_TO_TRI(j, i)];
        }
        for (j = i + 1; j < n_orb; j++) {
            s_tens[i] += d_same[I_J_TO_TRI(i, j)];
        }
        for (j = 0; j < n_orb; j++) {
            s_tens[i] += d_diff[i][j];
        }
    }
    hb_obj->s_tens = s_tens;
    
    double *exch_mat = malloc(n_orb * (n_orb - 1) / 2 * sizeof(double));
    tri_idx = 0;
    for (j = 0; j < n_orb; j++) {
        for (i = 0; i < j; i++) {
            exch_mat[tri_idx] = sqrt(fabs(eris[i + half_frz][j + half_frz][j + half_frz][i + half_frz]));
            tri_idx++;
        }
    }
    hb_obj->exch_sqrt = exch_mat;
    return hb_obj;
}


double calc_o1_probs(hb_info *tens, double *prob_arr, unsigned int n_elec,
                     unsigned char *occ_orbs) {
    double norm = 0;
    unsigned int orb_idx;
    for (orb_idx = 0; orb_idx < n_elec / 2; orb_idx++) {
        prob_arr[orb_idx] = tens->s_tens[occ_orbs[orb_idx]];
        norm += prob_arr[orb_idx];
    }
    for (orb_idx = n_elec / 2; orb_idx < n_elec; orb_idx++) {
        prob_arr[orb_idx] = tens->s_tens[occ_orbs[orb_idx] - tens->n_orb];
        norm += prob_arr[orb_idx];
    }
    double inv_norm = 1. / norm;
    for (orb_idx = 0; orb_idx < n_elec; orb_idx++) {
        prob_arr[orb_idx] *= inv_norm;
    }
    return norm;
}


double calc_o2_probs(hb_info *tens, double *prob_arr, unsigned int n_elec,
                     unsigned char *occ_orbs, unsigned char *o1) {
    double norm = 0;
    unsigned int orb_idx;
    unsigned char o1_orb = occ_orbs[*o1];
    size_t n_orb = tens->n_orb;
    int o1_spin = o1_orb / n_orb;
    
    double (*diff_tab)[n_orb] = (double (*)[n_orb])tens->d_diff;
//    if (o1_spin == 0) {
//        for (orb_idx = 0; orb_idx < *o1; orb_idx++) {
//            prob_arr[orb_idx] = tens->d_same[I_J_TO_TRI(occ_orbs[orb_idx], o1_orb)];
//            norm += prob_arr[orb_idx];
//        }
//        for (orb_idx = *o1 + 1; orb_idx < n_elec / 2; orb_idx++) {
//            prob_arr[orb_idx] = tens->d_same[I_J_TO_TRI(o1_orb, occ_orbs[orb_idx])];
//            norm += prob_arr[orb_idx];
//        }
//        for (orb_idx = n_elec / 2; orb_idx < n_elec; orb_idx++) {
//            prob_arr[orb_idx] = diff_tab[o1_orb][occ_orbs[orb_idx] - n_orb];
//            norm += prob_arr[orb_idx];
//        }
//    }
//    else {
//        for (orb_idx = 0; orb_idx < n_elec / 2; orb_idx++) {
//            prob_arr[orb_idx] = diff_tab[o1_orb % n_orb][occ_orbs[orb_idx]];
//            norm += prob_arr[orb_idx];
//        }
//        for (orb_idx = n_elec / 2; orb_idx < *o1; orb_idx++) {
//            prob_arr[orb_idx] = tens->d_same[I_J_TO_TRI(occ_orbs[orb_idx] - n_orb, o1_orb - n_orb)];
//            norm += prob_arr[orb_idx];
//        }
//        for (orb_idx = *o1 + 1; orb_idx < n_elec; orb_idx++) {
//            prob_arr[orb_idx] = tens->d_same[I_J_TO_TRI(o1_orb - n_orb, occ_orbs[orb_idx] - n_orb)];
//            norm += prob_arr[orb_idx];
//        }
//    }
    unsigned int offset = (1 - o1_spin) * n_elec / 2;
    for (orb_idx = offset; orb_idx < (n_elec / 2 + offset); orb_idx++) {
        prob_arr[orb_idx] = diff_tab[o1_orb % n_orb][occ_orbs[orb_idx] % n_orb];
        norm += prob_arr[orb_idx];
    }
    offset = o1_spin * n_elec / 2;
    for (orb_idx = offset; orb_idx < *o1; orb_idx++) {
        prob_arr[orb_idx] = tens->d_same[I_J_TO_TRI(occ_orbs[orb_idx] % n_orb, o1_orb % n_orb)];
        norm += prob_arr[orb_idx];
    }
    for (orb_idx = *o1 + 1; orb_idx < (n_elec / 2 + offset); orb_idx++) {
        prob_arr[orb_idx] = tens->d_same[I_J_TO_TRI(o1_orb % n_orb, occ_orbs[orb_idx] % n_orb)];
        norm += prob_arr[orb_idx];
    }
    prob_arr[*o1] = 0;
    
    double inv_norm = 1. / norm;
    for (orb_idx = 0; orb_idx < n_elec; orb_idx++) {
        prob_arr[orb_idx] *= inv_norm;
    }
    *o1 = o1_orb;
    return norm;
}


double calc_u1_probs(hb_info *tens, double *prob_arr, unsigned char o1_orb,
                     long long det) {
    unsigned int n_orb = (unsigned int)tens->n_orb;
    int o1_spin = o1_orb / n_orb;
    unsigned int o1_spinless = o1_orb % n_orb;
    unsigned int orb_idx;
    unsigned int offset = o1_spin * n_orb;
    double norm = 0;
    for (orb_idx = 0; orb_idx < o1_spinless; orb_idx++) {
        if (!((1LL << (orb_idx + offset)) & det)) {
            prob_arr[orb_idx] = tens->exch_sqrt[I_J_TO_TRI(orb_idx, o1_spinless)];
            norm += prob_arr[orb_idx];
        }
        else {
            prob_arr[orb_idx] = 0;
        }
    }
    prob_arr[o1_spinless] = 0;
    for (orb_idx = o1_spinless + 1; orb_idx < n_orb; orb_idx++) {
        if (!((1LL << (orb_idx + offset)) & det)) {
            prob_arr[orb_idx] = tens->exch_sqrt[I_J_TO_TRI(o1_spinless, orb_idx)];
            norm += prob_arr[orb_idx];
        }
        else {
            prob_arr[orb_idx] = 0;
        }
    }
    double inv_norm = 1. / norm;
    for (orb_idx = 0; orb_idx < n_orb; orb_idx++) {
        prob_arr[orb_idx] *= inv_norm;
    }
    return norm;
}


double calc_u2_probs(hb_info *tens, double *prob_arr, unsigned char o1_orb,
                     unsigned char o2_orb, unsigned char u1_orb,
                     unsigned char *lookup_mem, unsigned char *symm,
                     unsigned int prob_len) {
    unsigned int n_orb = (unsigned int)tens->n_orb;
    unsigned char (*lookup_tabl)[n_orb + 1] = (unsigned char (*)[n_orb + 1])lookup_mem;
    unsigned char o2_spinless = o2_orb % n_orb;
    unsigned char u1_spinless = u1_orb % n_orb;
    int same_spin = (o1_orb / n_orb) == (o2_orb / n_orb);
    unsigned char u2_irrep = symm[o1_orb % n_orb] ^ symm[o2_spinless] ^ symm[u1_spinless];
    unsigned int num_u2 = lookup_tabl[u2_irrep][0];
    unsigned int orb_idx;
    for (orb_idx = 0; orb_idx < prob_len; orb_idx++) {
        prob_arr[orb_idx] = 0;
    }
    unsigned char u2_orb;
    unsigned char min_o2_u2, max_o2_u2;
    double norm = 0;
    for (orb_idx = 0; orb_idx < num_u2; orb_idx++) {
        u2_orb = lookup_tabl[u2_irrep][orb_idx + 1];
        if ((same_spin && u2_orb != u1_spinless) || !same_spin) {
            min_o2_u2 = (o2_spinless < u2_orb) ? o2_spinless : u2_orb;
            max_o2_u2 = (o2_spinless > u2_orb) ? o2_spinless : u2_orb;
            prob_arr[orb_idx] = tens->exch_sqrt[I_J_TO_TRI(min_o2_u2, max_o2_u2)];
            norm += prob_arr[orb_idx];
        }
    }
    if (norm != 0) {
        double inv_norm = 1 / norm;
        for (orb_idx = 0; orb_idx < num_u2; orb_idx++) {
            u2_orb = lookup_tabl[u2_irrep][orb_idx + 1];
            if ((same_spin && u2_orb != u1_spinless) || !same_spin) {
                prob_arr[orb_idx] *= inv_norm;
            }
        }
    }
    return norm;
}

double calc_unnorm_wt(hb_info *tens, unsigned char *orbs) {
    unsigned int n_orb = (unsigned int)tens->n_orb;
    unsigned char o1 = orbs[0] % n_orb;
    unsigned char o2 = orbs[1] % n_orb;
    unsigned char u1 = orbs[2] % n_orb;
    unsigned char u2 = orbs[3] % n_orb;
    unsigned char min_o1_u1 = o1 < u1 ? o1 : u1;
    unsigned char max_o1_u1 = o1 > u1 ? o1 : u1;
    unsigned char min_o2_u2 = o2 < u2 ? o2 : u2;
    unsigned char max_o2_u2 = o2 > u2 ? o2 : u2;
    int same_sp = (orbs[0] / n_orb) == (orbs[1] / n_orb);
    double weight;
    if (same_sp) {
        unsigned char min_o1_u2 = o1 < u2 ? o1 : u2;
        unsigned char max_o1_u2 = o1 > u2 ? o1 : u2;
        unsigned char min_o2_u1 = o2 < u1 ? o2 : u1;
        unsigned char max_o2_u1 = o2 > u1 ? o2 : u1;
        weight = (tens->s_tens[o1] + tens->s_tens[o2]) * tens->d_same[I_J_TO_TRI(o1, o2)] * (tens->exch_sqrt[I_J_TO_TRI(min_o1_u1, max_o1_u1)] * tens->exch_sqrt[I_J_TO_TRI(min_o2_u2, max_o2_u2)] + tens->exch_sqrt[I_J_TO_TRI(min_o1_u2, max_o1_u2)] * tens->exch_sqrt[I_J_TO_TRI(min_o2_u1, max_o2_u1)]);
    }
    else {
        double (*diff_tab)[n_orb] = (double (*)[n_orb])tens->d_diff;
        weight = (tens->s_tens[o1] * diff_tab[o1][o2] + tens->s_tens[o2] * diff_tab[o2][o1]) * tens->exch_sqrt[I_J_TO_TRI(min_o1_u1, max_o1_u1)] * tens->exch_sqrt[I_J_TO_TRI(min_o2_u2, max_o2_u2)];
    }
    return weight;
}


double calc_norm_wt(hb_info *tens, unsigned char *orbs, unsigned char *occ, unsigned int n_elec, long long det) {
    unsigned int n_orb = (unsigned int)tens->n_orb;
    unsigned char o1 = orbs[0] % n_orb;
    int o1_spin = orbs[0] / n_orb;
    unsigned char o2 = orbs[1] % n_orb;
    int o2_spin = orbs[1] / n_orb;
    unsigned char u1 = orbs[2] % n_orb;
    unsigned char u2 = orbs[3] % n_orb;
    unsigned char min_o1_u1 = o1 < u1 ? o1 : u1;
    unsigned char max_o1_u1 = o1 > u1 ? o1 : u1;
    unsigned char min_o2_u2 = o2 < u2 ? o2 : u2;
    unsigned char max_o2_u2 = o2 > u2 ? o2 : u2;
    int same_sp = (orbs[0] / n_orb) == (orbs[1] / n_orb);
    double weight;
    size_t orb_idx;
    double s_denom = 0;
    for (orb_idx = 0; orb_idx < n_elec; orb_idx++) {
        s_denom += tens->s_tens[occ[orb_idx] % n_orb];
    }
    double (*diff_tab)[n_orb] = (double (*)[n_orb])tens->d_diff;
    double d1_denom = 0;
    unsigned int offset = (1 - o1_spin) * n_elec / 2;
    for (orb_idx = offset; orb_idx < (n_elec / 2 + offset); orb_idx++) {
        d1_denom += diff_tab[o1][occ[orb_idx] % n_orb];;
    }
    offset = o1_spin * n_elec / 2;
    for (orb_idx = offset; (occ[orb_idx] % n_orb) < o1; orb_idx++) {
        d1_denom += tens->d_same[I_J_TO_TRI(occ[orb_idx] % n_orb, o1)];
    }
    for (orb_idx++; orb_idx < (n_elec / 2 + offset); orb_idx++) {
        d1_denom += tens->d_same[I_J_TO_TRI(o1, occ[orb_idx] % n_orb)];
    }
    double d2_denom = 0;
    offset = (1 - o2_spin) * n_elec / 2;
    for (orb_idx = offset; orb_idx < (n_elec / 2 + offset); orb_idx++) {
        d2_denom += diff_tab[o2][occ[orb_idx] % n_orb];;
    }
    offset = o2_spin * n_elec / 2;
    for (orb_idx = offset; (occ[orb_idx] % n_orb) < o2; orb_idx++) {
        d2_denom += tens->d_same[I_J_TO_TRI(occ[orb_idx] % n_orb, o2)];
    }
    for (orb_idx++; orb_idx < (n_elec / 2 + offset); orb_idx++) {
        d2_denom += tens->d_same[I_J_TO_TRI(o2, occ[orb_idx] % n_orb)];
    }
    
    double e1_virt = 0;
    offset = o1_spin * n_orb;
    for (orb_idx = 0; orb_idx < o1; orb_idx++) {
        if (!((1LL << (orb_idx + offset)) & det)) {
            e1_virt += tens->exch_sqrt[I_J_TO_TRI(orb_idx, o1)];
        }
    }
    for (orb_idx = o1 + 1; orb_idx < n_orb; orb_idx++) {
        if (!((1LL << (orb_idx + offset)) & det)) {
            e1_virt += tens->exch_sqrt[I_J_TO_TRI(o1, orb_idx)];
        }
    }
    
    double e2_virt = 0;
    offset = o2_spin * n_orb;
    for (orb_idx = 0; orb_idx < o2; orb_idx++) {
        if (!((1LL << (orb_idx + offset)) & det)) {
            e2_virt += tens->exch_sqrt[I_J_TO_TRI(orb_idx, o2)];
        }
    }
    for (orb_idx = o2 + 1; orb_idx < n_orb; orb_idx++) {
        if (!((1LL << (orb_idx + offset)) & det)) {
            e2_virt += tens->exch_sqrt[I_J_TO_TRI(o2, orb_idx)];
        }
    }
    
    
    
    
    if (same_sp) {
        unsigned char min_o1_u2 = o1 < u2 ? o1 : u2;
        unsigned char max_o1_u2 = o1 > u2 ? o1 : u2;
        unsigned char min_o2_u1 = o2 < u1 ? o2 : u1;
        unsigned char max_o2_u1 = o2 > u1 ? o2 : u1;
        weight = (tens->s_tens[o1] + tens->s_tens[o2]) * tens->d_same[I_J_TO_TRI(o1, o2)] * (tens->exch_sqrt[I_J_TO_TRI(min_o1_u1, max_o1_u1)] * tens->exch_sqrt[I_J_TO_TRI(min_o2_u2, max_o2_u2)] + tens->exch_sqrt[I_J_TO_TRI(min_o1_u2, max_o1_u2)] * tens->exch_sqrt[I_J_TO_TRI(min_o2_u1, max_o2_u1)]);
    }
    else {
        double (*diff_tab)[n_orb] = (double (*)[n_orb])tens->d_diff;
        weight = (tens->s_tens[o1] * diff_tab[o1][o2] + tens->s_tens[o2] * diff_tab[o2][o1]) * tens->exch_sqrt[I_J_TO_TRI(min_o1_u1, max_o1_u1)] * tens->exch_sqrt[I_J_TO_TRI(min_o2_u2, max_o2_u2)];
    }
    return weight;
}
