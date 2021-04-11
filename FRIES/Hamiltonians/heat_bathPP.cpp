/*! \file
 *
 * \brief Utilities for Heat-Bath Power-Pitzer compression of Hamiltonian
 *
 * These functions apply only to the double excitation portion of a molecular
 * Hamiltonian matrix. Single excitations are treated as in the near-uniform
 * scheme
 */

#include "heat_bathPP.hpp"
#include <FRIES/det_store.h>
#include <FRIES/Hamiltonians/molecule.hpp>

#define TRI_N(n)((n) * (n + 1) / 2)
#define I_J_TO_TRI(i, j)(TRI_N(j - 1) + i)

hb_info *set_up(unsigned int tot_orb, unsigned int n_orb,
                const FourDArr &eris) {
    hb_info *hb_obj = (hb_info *)malloc(sizeof(hb_info));
    hb_obj->n_orb = n_orb;
    unsigned int half_frz = tot_orb - n_orb;
    
    double *d_diff = (double *)calloc(n_orb * n_orb, sizeof(double));
    size_t i, j, a, b;
    for (i = 0; i < n_orb; i++) {
        for (j = 0; j < n_orb; j++) {
            for (a = half_frz; a < tot_orb; a++) {
                for (b = half_frz; b < tot_orb; b++) {
                    if (i != (a - half_frz) && j != (b - half_frz)) {
                        d_diff[i * n_orb + j] += fabs(eris(i + half_frz, j + half_frz, a, b)); // exchange terms are zero
                    }
                }
            }
        }
    }
    hb_obj->d_diff = d_diff;
    
    double *d_same = (double *)calloc(n_orb * (n_orb - 1) / 2, sizeof(double));
    size_t tri_idx = 0;
    for (j = 1; j < n_orb; j++) {
        for (i = 0; i < j; i++) {
            for (a = half_frz; a < tot_orb; a++) {
                for (b = half_frz; b < a; b++) {
                    if ((a - half_frz) != j && (a - half_frz) != i && (b - half_frz) != j && (b - half_frz) != i) {
                        d_same[tri_idx] += 2 * fabs(eris(i + half_frz, j + half_frz, a, b) - eris(i + half_frz, j + half_frz, b, a));
                    }
                }
            }
            tri_idx++;
        }
    }
    hb_obj->d_same = d_same;
    
    double *s_tens = (double *)calloc(n_orb, sizeof(double));
    hb_obj->s_norm = 0;
    for (i = 0; i < n_orb; i++) {
        for (j = 0; j < i; j++) {
            s_tens[i] += d_same[I_J_TO_TRI(j, i)];
        }
        for (j = i + 1; j < n_orb; j++) {
            s_tens[i] += d_same[I_J_TO_TRI(i, j)];
        }
        for (j = 0; j < n_orb; j++) {
            s_tens[i] += d_diff[i * n_orb + j];
        }
        hb_obj->s_norm += s_tens[i];
    }
    hb_obj->s_tens = s_tens;
    
    double *exch_mat = (double *)malloc(n_orb * (n_orb - 1) / 2 * sizeof(double));
    tri_idx = 0;
    for (j = 0; j < n_orb; j++) {
        for (i = 0; i < j; i++) {
            exch_mat[tri_idx] = sqrt(fabs(eris(i + half_frz, j + half_frz, j + half_frz, i + half_frz)));
            tri_idx++;
        }
    }
    hb_obj->exch_sqrt = exch_mat;
    
    double *diag_part = (double *)malloc(sizeof(double) * n_orb);
    for (j = 0; j < n_orb; j++) {
        diag_part[j] = sqrt(fabs(eris(j + half_frz, j + half_frz, j + half_frz, j + half_frz)));
    }
    hb_obj->diag_sqrt = diag_part;
    
    double *exch_norms = (double *)calloc(n_orb, sizeof(double));
    for (i = 0; i < n_orb; i++) {
        for (j = 0; j < i; j++) {
            exch_norms[i] += exch_mat[I_J_TO_TRI(j, i)];
        }
        exch_norms[i] += diag_part[i];
        for (j = i + 1; j < n_orb; j++) {
            exch_norms[i] += exch_mat[I_J_TO_TRI(i, j)];
        }
    }
    hb_obj->exch_norms = exch_norms;
    return hb_obj;
}


double calc_o1_probs(hb_info *tens, double *prob_arr, unsigned int n_elec,
                     uint8_t *occ_orbs, int exclude_first) {
    double norm = 0;
    uint8_t skip_first = exclude_first > 0;
    for (unsigned int orb_idx = skip_first; orb_idx < n_elec / 2; orb_idx++) {
        prob_arr[orb_idx - skip_first] = tens->s_tens[occ_orbs[orb_idx]];
        norm += prob_arr[orb_idx - skip_first];
    }
    for (unsigned int orb_idx = n_elec / 2; orb_idx < n_elec; orb_idx++) {
        prob_arr[orb_idx - skip_first] = tens->s_tens[occ_orbs[orb_idx] - tens->n_orb];
        norm += prob_arr[orb_idx - skip_first];
    }
    double inv_norm = 1. / norm;
    for (unsigned int orb_idx = skip_first; orb_idx < n_elec; orb_idx++) {
        prob_arr[orb_idx - skip_first] *= inv_norm;
    }
    norm /= tens->s_norm;
    return norm;
}


double calc_o2_probs(hb_info *tens, double *prob_arr, unsigned int n_elec,
                     uint8_t *occ_orbs, uint8_t o1_idx) {
    double norm = 0;
    uint8_t o1_orb = occ_orbs[o1_idx];
    size_t n_orb = tens->n_orb;
    int o1_spin = o1_orb / n_orb;
    
    double *diff_tab = tens->d_diff;
    unsigned int offset = (1 - o1_spin) * n_elec / 2;
    for (unsigned int orb_idx = offset; orb_idx < (n_elec / 2 + offset); orb_idx++) {
        prob_arr[orb_idx] = diff_tab[(o1_orb % n_orb) * n_orb + occ_orbs[orb_idx] % n_orb];
        norm += prob_arr[orb_idx];
    }
    offset = o1_spin * n_elec / 2;
    for (unsigned int orb_idx = offset; orb_idx < o1_idx; orb_idx++) {
        prob_arr[orb_idx] = tens->d_same[I_J_TO_TRI(occ_orbs[orb_idx] % n_orb, o1_orb % n_orb)];
        norm += prob_arr[orb_idx];
    }
    for (unsigned int orb_idx = o1_idx + 1; orb_idx < (n_elec / 2 + offset); orb_idx++) {
        prob_arr[orb_idx] = tens->d_same[I_J_TO_TRI(o1_orb % n_orb, occ_orbs[orb_idx] % n_orb)];
        norm += prob_arr[orb_idx];
    }
    prob_arr[o1_idx] = 0;
    
    double inv_norm = 1. / norm;
    for (unsigned int orb_idx = 0; orb_idx < n_elec; orb_idx++) {
        prob_arr[orb_idx] *= inv_norm;
    }
    norm /= tens->s_tens[o1_orb % n_orb];
    return norm;
}


double calc_o2_probs_half(hb_info *tens, double *prob_arr, unsigned int n_elec,
                          uint8_t *occ_orbs, uint8_t o1_idx) {
    double norm = 0;
    uint8_t o1_orb = occ_orbs[o1_idx];
    size_t n_orb = tens->n_orb;
    int o1_spin = o1_orb / n_orb;
    
    double *diff_tab = tens->d_diff;
    unsigned int upper_limit = n_elec / 2 > o1_idx ? o1_idx : n_elec / 2;
    for (unsigned int orb_idx = 0; orb_idx < upper_limit; orb_idx++) {
        if (o1_spin == 0) {
            prob_arr[orb_idx] = tens->d_same[I_J_TO_TRI(occ_orbs[orb_idx], o1_orb)];
        }
        else {
            prob_arr[orb_idx] = diff_tab[(o1_orb - n_orb) * n_orb + occ_orbs[orb_idx]];
        }
        norm += prob_arr[orb_idx];
    }
    for (unsigned int orb_idx = n_elec / 2; orb_idx < o1_idx; orb_idx++) {
        if (o1_spin == 0) {
            prob_arr[orb_idx] = diff_tab[o1_orb * n_orb + occ_orbs[orb_idx] - n_orb];
        }
        else {
            prob_arr[orb_idx] = tens->d_same[I_J_TO_TRI(occ_orbs[orb_idx] - n_orb, o1_orb - n_orb)];
        }
        norm += prob_arr[orb_idx];
    }
    
    double inv_norm = 1. / norm;
    for (unsigned int orb_idx = 0; orb_idx < o1_idx; orb_idx++) {
        prob_arr[orb_idx] *= inv_norm;
    }
    norm /= tens->s_tens[o1_orb % n_orb];
    return norm;
}


double calc_u1_probs(hb_info *tens, double *prob_arr, uint8_t o1_orb,
                     uint8_t *occ_orbs, uint8_t n_elec, int exclude_first) {
    unsigned int n_orb = (unsigned int)tens->n_orb;
    int o1_spin = o1_orb / n_orb;
    unsigned int o1_spinless = o1_orb % n_orb;
    unsigned int offset = o1_spin * n_orb;
    double norm = 0;
    size_t prob_idx = 0;
    uint8_t occ_idx = n_elec / 2 * o1_spin;
    uint8_t curr_occ = occ_orbs[occ_idx];
    for (unsigned int orb_idx = 0; orb_idx < o1_spinless; orb_idx++) {
        if (orb_idx + offset == curr_occ) {
            occ_idx++;
            curr_occ = occ_orbs[occ_idx];
        }
        else {
            prob_arr[prob_idx] = tens->exch_sqrt[I_J_TO_TRI(orb_idx, o1_spinless)];
            norm += prob_arr[prob_idx];
            prob_idx++;
        }
    }
    occ_idx++;
    curr_occ = occ_orbs[occ_idx];
    for (unsigned int orb_idx = o1_spinless + 1; orb_idx < n_orb; orb_idx++) {
        if (orb_idx + offset == curr_occ) {
            if (occ_idx < n_elec - 1) {
                occ_idx++;
                curr_occ = occ_orbs[occ_idx];
            }
        }
        else {
            prob_arr[prob_idx] = tens->exch_sqrt[I_J_TO_TRI(o1_spinless, orb_idx)];
            norm += prob_arr[prob_idx];
            prob_idx++;
        }
    }
    if (exclude_first) {
        norm -= prob_arr[0];
        prob_arr[0] = 0;
    }
    double inv_norm = 1. / norm;
    for (unsigned int orb_idx = 0; orb_idx < prob_idx; orb_idx++) {
        prob_arr[orb_idx] *= inv_norm;
    }
    norm /= tens->exch_norms[o1_spinless];
    return norm;
}


double calc_u2_probs(hb_info *tens, double *prob_arr, uint8_t o1_orb,
                     uint8_t o2_orb, uint8_t u1_orb,
                     SymmInfo *symm_struct, uint16_t *prob_len) {
    unsigned int n_orb = (unsigned int)tens->n_orb;
    uint8_t o2_spinless = o2_orb % n_orb;
    uint8_t u1_spinless = u1_orb % n_orb;
    int same_spin = (o1_orb / n_orb) == (o2_orb / n_orb);
    std::vector<uint8_t> &symm = symm_struct->symm_vec;
    uint8_t u2_irrep = symm[o1_orb % n_orb] ^ symm[o2_spinless] ^ symm[u1_spinless];
    unsigned int num_u2 = symm_struct->symm_lookup(u2_irrep, 0);
    *prob_len = num_u2;
    
    uint8_t u2_orb;
    uint8_t min_o2_u2, max_o2_u2;
    double norm = 0;
    for (unsigned int orb_idx = 0; orb_idx < num_u2; orb_idx++) {
        u2_orb = symm_struct->symm_lookup(u2_irrep, orb_idx + 1);
        if ((same_spin && u2_orb != u1_spinless) || !same_spin) {
            if (o2_spinless == u2_orb) {
                prob_arr[orb_idx] = tens->diag_sqrt[o2_spinless];
            }
            else {
                min_o2_u2 = (o2_spinless < u2_orb) ? o2_spinless : u2_orb;
                max_o2_u2 = (o2_spinless > u2_orb) ? o2_spinless : u2_orb;
                prob_arr[orb_idx] = tens->exch_sqrt[I_J_TO_TRI(min_o2_u2, max_o2_u2)];
            }
            norm += prob_arr[orb_idx];
        }
        else {
            prob_arr[orb_idx] = 0;
        }
    }
    if (norm != 0) {
        double inv_norm = 1 / norm;
        for (unsigned int orb_idx = 0; orb_idx < num_u2; orb_idx++) {
            u2_orb = symm_struct->symm_lookup(u2_irrep, orb_idx + 1);
            if ((same_spin && u2_orb != u1_spinless) || !same_spin) {
                prob_arr[orb_idx] *= inv_norm;
            }
        }
    }
    norm /= tens->exch_norms[o2_spinless];
    return norm;
}


double calc_u2_probs_half(hb_info *tens, double *prob_arr, uint8_t o1_orb,
                          uint8_t o2_orb, uint8_t u1_orb, uint8_t *det,
                          SymmInfo *symm_struct, uint16_t *prob_len) {
    unsigned int n_orb = (unsigned int)tens->n_orb;
    uint8_t o2_spinless = o2_orb % n_orb;
    uint8_t u1_spinless = u1_orb % n_orb;
    int u2_spin = o2_orb / n_orb;
    int same_spin = (o1_orb / n_orb) == u2_spin;
    std::vector<uint8_t> &symm = symm_struct->symm_vec;
    uint8_t u2_irrep = symm[o1_orb % n_orb] ^ symm[o2_spinless] ^ symm[u1_spinless];
    unsigned int num_u2 = symm_struct->symm_lookup(u2_irrep, 0);
    
    uint8_t u2_orb, orb_idx;
    uint8_t min_o2_u2, max_o2_u2;
    double norm = 0;
    for (orb_idx = 0; orb_idx < num_u2; orb_idx++) {
        u2_orb = symm_struct->symm_lookup(u2_irrep, orb_idx + 1);
        if (same_spin && u2_orb >= u1_spinless) {
            break;
        }
        if (((same_spin && u2_orb != u1_spinless) || !same_spin) && !read_bit(det, u2_orb + n_orb * u2_spin)) {
            if (o2_spinless == u2_orb) {
                prob_arr[orb_idx] = tens->diag_sqrt[o2_spinless];
            }
            else {
                min_o2_u2 = (o2_spinless < u2_orb) ? o2_spinless : u2_orb;
                max_o2_u2 = (o2_spinless > u2_orb) ? o2_spinless : u2_orb;
                prob_arr[orb_idx] = tens->exch_sqrt[I_J_TO_TRI(min_o2_u2, max_o2_u2)];
            }
            norm += prob_arr[orb_idx];
        }
        else {
            prob_arr[orb_idx] = 0;
        }
    }
    *prob_len = orb_idx;
    if (norm != 0) {
        double inv_norm = 1 / norm;
        for (orb_idx = 0; orb_idx < *prob_len; orb_idx++) {
            prob_arr[orb_idx] *= inv_norm;
        }
    }
    norm /= tens->exch_norms[o2_spinless];
    return norm;
}

double calc_unnorm_wt(hb_info *tens, uint8_t *orbs) {
    unsigned int n_orb = (unsigned int)tens->n_orb;
    uint8_t o1 = orbs[0] % n_orb;
    uint8_t o2 = orbs[1] % n_orb;
    uint8_t u1 = orbs[2] % n_orb;
    uint8_t u2 = orbs[3] % n_orb;
    uint8_t min_o1_u1 = o1 < u1 ? o1 : u1;
    uint8_t max_o1_u1 = o1 > u1 ? o1 : u1;
    uint8_t min_o2_u2 = o2 < u2 ? o2 : u2;
    uint8_t max_o2_u2 = o2 > u2 ? o2 : u2;
    int same_sp = (orbs[0] / n_orb) == (orbs[1] / n_orb);
    double weight;
    if (same_sp) {
        weight = tens->d_same[I_J_TO_TRI(o1, o2)] * (tens->exch_sqrt[I_J_TO_TRI(min_o1_u1, max_o1_u1)] * tens->exch_sqrt[I_J_TO_TRI(min_o2_u2, max_o2_u2)]) / tens->s_norm / tens->exch_norms[o1] / tens->exch_norms[o2];
    }
    else {
        double *diff_tab = tens->d_diff;
        weight = (diff_tab[o1 * n_orb + o2]) * tens->exch_sqrt[I_J_TO_TRI(min_o1_u1, max_o1_u1)] * tens->exch_sqrt[I_J_TO_TRI(min_o2_u2, max_o2_u2)] / tens->s_norm / tens->exch_norms[o1] / tens->exch_norms[o2];
    }
    return weight;
}


double calc_norm_wt(hb_info *tens, uint8_t *orbs, uint8_t *occ,
                    unsigned int n_elec, uint8_t *det,
                    SymmInfo *symm) {
    unsigned int n_orb = (unsigned int)tens->n_orb;
    uint8_t o1 = orbs[0] % n_orb;
    int o1_spin = orbs[0] / n_orb;
    uint8_t o2 = orbs[1] % n_orb;
    int o2_spin = orbs[1] / n_orb;
    uint8_t u1 = orbs[2] % n_orb;
    uint8_t u2 = orbs[3] % n_orb;
    uint8_t min_o1_u1 = o1 < u1 ? o1 : u1;
    uint8_t max_o1_u1 = o1 > u1 ? o1 : u1;
    uint8_t min_o2_u2 = o2 < u2 ? o2 : u2;
    uint8_t max_o2_u2 = o2 > u2 ? o2 : u2;
    int same_sp = (orbs[0] / n_orb) == (orbs[1] / n_orb);
    size_t orb_idx;
    
    Matrix<uint8_t> &lookup_tabl = symm->symm_lookup;
    
    uint8_t occ_spatial[n_elec];
    for (orb_idx = 0; orb_idx < n_elec; orb_idx++) {
        occ_spatial[orb_idx] = occ[orb_idx] % n_orb;
    }
    
    double weight;
    double s_denom = 0;
    for (orb_idx = 0; orb_idx < n_elec; orb_idx++) {
        s_denom += tens->s_tens[occ_spatial[orb_idx]];
    }
    double *diff_tab = tens->d_diff;
    double d1_denom = 0;
    unsigned int offset = (1 - o1_spin) * n_elec / 2;
    for (orb_idx = offset; orb_idx < (n_elec / 2 + offset); orb_idx++) {
        d1_denom += diff_tab[o1 * n_orb + occ_spatial[orb_idx]];
    }
    offset = o1_spin * n_elec / 2;
    for (orb_idx = offset; occ_spatial[orb_idx] < o1; orb_idx++) {
        d1_denom += tens->d_same[I_J_TO_TRI(occ_spatial[orb_idx], o1)];
    }
    for (orb_idx++; orb_idx < (n_elec / 2 + offset); orb_idx++) {
        d1_denom += tens->d_same[I_J_TO_TRI(o1, occ_spatial[orb_idx])];
    }
    double d2_denom = 0;
    offset = (1 - o2_spin) * n_elec / 2;
    for (orb_idx = offset; orb_idx < (n_elec / 2 + offset); orb_idx++) {
        d2_denom += diff_tab[o2 * n_orb + occ_spatial[orb_idx]];;
    }
    offset = o2_spin * n_elec / 2;
    for (orb_idx = offset; occ_spatial[orb_idx] < o2; orb_idx++) {
        d2_denom += tens->d_same[I_J_TO_TRI(occ_spatial[orb_idx], o2)];
    }
    for (orb_idx++; orb_idx < (n_elec / 2 + offset); orb_idx++) {
        d2_denom += tens->d_same[I_J_TO_TRI(o2, occ_spatial[orb_idx])];
    }
    
    double e1_virt = 0;
    offset = o1_spin * n_orb;
    for (orb_idx = 0; orb_idx < o1; orb_idx++) {
        if (!read_bit(det, orb_idx + offset)) {
            e1_virt += tens->exch_sqrt[I_J_TO_TRI(orb_idx, o1)];
        }
    }
    for (orb_idx = o1 + 1; orb_idx < n_orb; orb_idx++) {
        if (!read_bit(det, orb_idx + offset)) {
            e1_virt += tens->exch_sqrt[I_J_TO_TRI(o1, orb_idx)];
        }
    }
    
    double e2_virt = 0;
    offset = o2_spin * n_orb;
    for (orb_idx = 0; orb_idx < o2; orb_idx++) {
        if (!read_bit(det, orb_idx + offset)) {
            e2_virt += tens->exch_sqrt[I_J_TO_TRI(orb_idx, o2)];
        }
    }
    for (orb_idx = o2 + 1; orb_idx < n_orb; orb_idx++) {
        if (!read_bit(det, orb_idx + offset)) {
            e2_virt += tens->exch_sqrt[I_J_TO_TRI(o2, orb_idx)];
        }
    }
    
    uint8_t u1_irrep = symm->symm_vec[u1];
    uint8_t u2_irrep = symm->symm_vec[u2];
    uint8_t symm_orb, min_orb, max_orb;
    
    double e2_symm_no1 = 0;
    double e2_symm_no2 = 0;
    double e1_symm_no1 = 0;
    double e1_symm_no2 = 0;
    for (orb_idx = 0; orb_idx < lookup_tabl(u2_irrep, 0); orb_idx++) {
        symm_orb = lookup_tabl(u2_irrep, orb_idx + 1);
        if ((same_sp && symm_orb != u1) || !same_sp) {
            if (o2 == symm_orb) {
                e2_symm_no1 += tens->diag_sqrt[o2];
            }
            else {
                min_orb = (o2 < symm_orb) ? o2 : symm_orb;
                max_orb = (o2 > symm_orb) ? o2 : symm_orb;
                e2_symm_no1 += tens->exch_sqrt[I_J_TO_TRI(min_orb, max_orb)];
            }
        }
        if ((same_sp && symm_orb != u1) || !same_sp) {
            if (o1 == symm_orb) {
                e1_symm_no1 += tens->diag_sqrt[o1];
            }
            else {
                min_orb = (o1 < symm_orb) ? o1 : symm_orb;
                max_orb = (o1 > symm_orb) ? o1 : symm_orb;
                e1_symm_no1 += tens->exch_sqrt[I_J_TO_TRI(min_orb, max_orb)];
            }
        }
    }
    
    for (orb_idx = 0; orb_idx < lookup_tabl(u1_irrep, 0); orb_idx++) {
        symm_orb = lookup_tabl(u1_irrep, orb_idx + 1);
        if ((same_sp && symm_orb != u2) || !same_sp) {
            if (o2 == symm_orb) {
                e2_symm_no2 += tens->diag_sqrt[o2];
            }
            else {
                min_orb = (o2 < symm_orb) ? o2 : symm_orb;
                max_orb = (o2 > symm_orb) ? o2 : symm_orb;
                e2_symm_no2 += tens->exch_sqrt[I_J_TO_TRI(min_orb, max_orb)];
            }
        }
        if ((same_sp && symm_orb != u2) || !same_sp) {
            if (o1 == symm_orb) {
                e1_symm_no2 += tens->diag_sqrt[o1];
            }
            else {
                min_orb = (o1 < symm_orb) ? o1 : symm_orb;
                max_orb = (o1 > symm_orb) ? o1 : symm_orb;
                e1_symm_no2 += tens->exch_sqrt[I_J_TO_TRI(min_orb, max_orb)];
            }
        }
    }
    
    unsigned int o1u1tri = I_J_TO_TRI(min_o1_u1, max_o1_u1);
    unsigned int o2u2tri = I_J_TO_TRI(min_o2_u2, max_o2_u2);
    if (same_sp) {
        uint8_t min_o1_u2 = o1 < u2 ? o1 : u2;
        uint8_t max_o1_u2 = o1 > u2 ? o1 : u2;
        uint8_t min_o2_u1 = o2 < u1 ? o2 : u1;
        uint8_t max_o2_u1 = o2 > u1 ? o2 : u1;
        unsigned int o1o2tri = I_J_TO_TRI(o1, o2);
        unsigned int o1u2tri = I_J_TO_TRI(min_o1_u2, max_o1_u2);
        unsigned int o2u1tri = I_J_TO_TRI(min_o2_u1, max_o2_u1);
        weight = tens->d_same[o1o2tri] / s_denom * (
        tens->s_tens[o1] / d1_denom / e1_virt * (tens->exch_sqrt[o1u1tri] * tens->exch_sqrt[o2u2tri] / e2_symm_no1 + tens->exch_sqrt[o1u2tri] * tens->exch_sqrt[o2u1tri] / e2_symm_no2) +
        tens->s_tens[o2] / d2_denom / e2_virt * (tens->exch_sqrt[o2u1tri] * tens->exch_sqrt[o1u2tri] / e1_symm_no1 + tens->exch_sqrt[o2u2tri] * tens->exch_sqrt[o1u1tri] / e1_symm_no2));
    }
    else {
        double *diff_tab = tens->d_diff;
        weight = (tens->s_tens[o1] * diff_tab[o1 * n_orb + o2] / d1_denom / e1_virt / e2_symm_no1 + tens->s_tens[o2] * diff_tab[o2 * n_orb + o1] / d2_denom / e2_virt / e1_symm_no2) * tens->exch_sqrt[o1u1tri] * tens->exch_sqrt[o2u2tri] / s_denom;
    }
    return weight;
}


unsigned int hb_doub_multi(uint8_t *det, uint8_t *occ_orbs,
                           unsigned int num_elec, SymmInfo *symm,
                           hb_info *tens,
                           unsigned int num_sampl, std::mt19937 &mt_obj,
                           uint8_t (* chosen_orbs)[4], double *prob_vec) {
    unsigned int num_orb = (unsigned int) tens->n_orb;
    unsigned int n_virt = num_orb - num_elec / 2;
    unsigned int n_states = num_elec > n_virt ? num_elec : n_virt;
    unsigned int alias_idx[n_states];
    double alias_probs[n_states];
    double probs[n_states];
    std::vector<uint8_t> &orb_symm = symm->symm_vec;
    
    // Choose first occupied orbital
    calc_o1_probs(tens, probs, num_elec, occ_orbs, 0);
    setup_alias(probs, alias_idx, alias_probs, num_elec);
    sample_alias(alias_idx, alias_probs, num_elec, chosen_orbs[0], num_sampl, 4, mt_obj);
    
    unsigned int o1_samples[num_elec];
    for (unsigned int elec_idx = 0; elec_idx < num_elec; elec_idx++) {
        o1_samples[elec_idx] = 0;
    }
    for (unsigned int samp_idx = 0; samp_idx < num_sampl; samp_idx++) {
        o1_samples[chosen_orbs[samp_idx][0]]++;
    }
    
    unsigned int loc_n_samp;
    unsigned int tot_sampled = 0;
    for (unsigned int elec_idx = 0; elec_idx < num_elec; elec_idx++) {
        loc_n_samp = o1_samples[elec_idx];
        if (loc_n_samp == 0) {
            continue;
        }
        uint8_t o1 = elec_idx;
        unsigned int samp_begin = tot_sampled;
        // Choose second occupied orbital
        calc_o2_probs(tens, probs, num_elec, occ_orbs, o1);
        o1 = occ_orbs[o1];
        setup_alias(probs, alias_idx, alias_probs, num_elec);
        sample_alias(alias_idx, alias_probs, num_elec, &chosen_orbs[samp_begin][1], loc_n_samp, 4, mt_obj);
        
        // Choose first virtual orbital
        calc_u1_probs(tens, probs, o1, occ_orbs, num_elec, 0);
        setup_alias(probs, alias_idx, alias_probs, n_virt);
        sample_alias(alias_idx, alias_probs, n_virt, &chosen_orbs[samp_begin][2], loc_n_samp, 4, mt_obj);
        
        for (unsigned int samp_idx = samp_begin; samp_idx < samp_begin + loc_n_samp; samp_idx++) {
            uint8_t o2 = occ_orbs[chosen_orbs[samp_idx][1]];
            uint8_t u1 = find_nth_virt(occ_orbs, o1 / num_orb, num_elec, num_orb, chosen_orbs[samp_idx][2]);
            uint8_t u2_symm = orb_symm[o1 % num_orb] ^ orb_symm[o2 % num_orb] ^ orb_symm[u1 % num_orb];
            uint16_t num_u2 = 0;
            double u2_norm = calc_u2_probs(tens, probs, o1, o2, u1, symm, &num_u2);
            if (u2_norm != 0) {
                setup_alias(probs, alias_idx, alias_probs, num_u2);
                uint8_t u2;
                sample_alias(alias_idx, alias_probs, num_u2, &u2, 1, 1, mt_obj);
                u2 = symm->symm_lookup(u2_symm, u2 + 1) + num_orb * (o2 / num_orb);
                if (read_bit(det, u2)) {
                    continue;
                }
                if (o1 > o2) {
                    chosen_orbs[tot_sampled][0] = o2;
                    chosen_orbs[tot_sampled][1] = o1;
                }
                else {
                    chosen_orbs[tot_sampled][0] = o1;
                    chosen_orbs[tot_sampled][1] = o2;
                }
                if (u1 > u2) {
                    chosen_orbs[tot_sampled][2] = u2;
                    chosen_orbs[tot_sampled][3] = u1;
                }
                else {
                    chosen_orbs[tot_sampled][2] = u1;
                    chosen_orbs[tot_sampled][3] = u2;
                }
                prob_vec[tot_sampled] = calc_norm_wt(tens, &chosen_orbs[tot_sampled][0], occ_orbs, num_elec, det, symm);
                tot_sampled++;
            }
        }
    }
    return tot_sampled;
}

// copy values to vec1
// initialize det_indices1
// n_samp = matr_samp - tot_dense_h
void apply_HBPP(Matrix<uint8_t> &all_orbs, Matrix<uint8_t> &all_dets, HBCompress *comp_scratch,
                hb_info *hb_probs, SymmInfo *symm,
                double p_doub, bool new_hb, std::mt19937 &mt_obj, uint32_t n_samp) {
    std::vector<double> &vec1 = comp_scratch->vec1;
    std::vector<double> &vec2 = comp_scratch->vec2;
    Matrix<double> &subwts = comp_scratch->subwts;
    Matrix<bool> &keep_sub = comp_scratch->keep_sub;
    std::vector<uint32_t> &ndiv = comp_scratch->ndiv;
    std::vector<uint16_t> &nsub = comp_scratch->nsub;
    size_t comp_len = comp_scratch->vec_len;
    std::vector<size_t> &det_indices1 = comp_scratch->det_indices1;
    std::vector<size_t> &det_indices2 = comp_scratch->det_indices2;
    uint8_t (*orb_indices1)[4] = comp_scratch->orb_indices1;
    uint8_t (*orb_indices2)[4] = comp_scratch->orb_indices2;
    size_t (*comp_idx)[2] = comp_scratch->comp_idx;
    std::vector<double> &wt_remain = comp_scratch->wt_remain;
    
    size_t spawn_length = vec1.size();
    int proc_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    double rn_sys = 0;
    uint32_t n_elec = (uint32_t) all_orbs.cols();
    uint32_t n_orb = (uint32_t) hb_probs->n_orb;
    unsigned int unocc_symm_cts[n_irreps][2];
    
#pragma mark Singles vs doubles
    subwts.reshape(spawn_length, 2);
    keep_sub.reshape(spawn_length, 2);
    for (size_t det_idx = 0; det_idx < comp_len; det_idx++) {
        double weight = fabs(vec1[det_idx]);
        vec1[det_idx] = weight;
        if (weight > 0) {
            subwts(det_idx, 0) = p_doub;
            subwts(det_idx, 1) = 1 - p_doub;
            ndiv[det_idx] = 0;
        }
        else {
            ndiv[det_idx] = 1;
        }
    }
    if (proc_rank == 0) {
        rn_sys = mt_obj() / (1. + UINT32_MAX);
    }
    comp_len = comp_sub(vec1.data(), comp_len, ndiv.data(), subwts, keep_sub, NULL, n_samp, wt_remain.data(), rn_sys, vec2.data(), comp_idx);
    if (comp_len > spawn_length) {
        std::cerr << "Error: insufficient memory allocated for matrix compression.\n";
    }
                
#pragma mark  First occupied orbital
    subwts.reshape(spawn_length, n_elec - new_hb);
    keep_sub.reshape(spawn_length, n_elec - new_hb);
    for (size_t samp_idx = 0; samp_idx < comp_len; samp_idx++) {
        size_t det_idx = det_indices1[comp_idx[samp_idx][0]];
        det_indices2[samp_idx] = det_idx;
        orb_indices1[samp_idx][0] = comp_idx[samp_idx][1];
        uint8_t *occ_orbs = all_orbs[det_idx];
        if (orb_indices1[samp_idx][0] == 0) { // double excitation
            ndiv[samp_idx] = 0;
            double tot_weight = calc_o1_probs(hb_probs, subwts[samp_idx], n_elec, occ_orbs, new_hb);
            if (new_hb) {
                vec2[samp_idx] *= tot_weight;
            }
        }
        else {
            count_symm_virt(unocc_symm_cts, occ_orbs, n_elec, symm);
            unsigned int n_occ = count_sing_allowed(occ_orbs, n_elec, symm->symm_vec.data(), n_orb, unocc_symm_cts);
            if (n_occ == 0) {
                ndiv[samp_idx] = 1;
                vec2[samp_idx] = 0;
            }
            else {
                ndiv[samp_idx] = n_occ;
            }
        }
    }
    
    if (proc_rank == 0) {
        rn_sys = mt_obj() / (1. + UINT32_MAX);
    }
    comp_len = comp_sub(vec2.data(), comp_len, ndiv.data(), subwts, keep_sub, NULL, n_samp, wt_remain.data(), rn_sys, vec1.data(), comp_idx);
    if (comp_len > spawn_length) {
        std::cerr << "Error: insufficient memory allocated for matrix compression.\n";
    }
                
#pragma mark Unoccupied orbital (single); 2nd occupied (double)
    for (size_t samp_idx = 0; samp_idx < comp_len; samp_idx++) {
        size_t weight_idx = comp_idx[samp_idx][0];
        size_t det_idx = det_indices2[weight_idx];
        det_indices1[samp_idx] = det_idx;
        orb_indices2[samp_idx][0] = orb_indices1[weight_idx][0]; // single or double
        orb_indices2[samp_idx][1] = comp_idx[samp_idx][1]; // first occupied orbital index (NOT converted to orbital below)
        if (orb_indices2[samp_idx][1] >= n_elec) {
            std::cerr << "Error: chosen occupied orbital (first) is out of bounds\n";
            vec1[samp_idx] = 0;
            ndiv[samp_idx] = 1;
            continue;
        }
        uint8_t *occ_orbs = all_orbs[det_idx];
        if (orb_indices2[samp_idx][0] == 0) { // double excitation
            ndiv[samp_idx] = 0;
            if (new_hb) {
                orb_indices2[samp_idx][1]++;
                nsub[samp_idx] = orb_indices2[samp_idx][1];
                vec1[samp_idx] *= calc_o2_probs_half(hb_probs, subwts[samp_idx], n_elec, occ_orbs, orb_indices2[samp_idx][1]);
            }
            else {
                calc_o2_probs(hb_probs, subwts[samp_idx], n_elec, occ_orbs, orb_indices2[samp_idx][1]);
            }
        }
        else { // single excitation
            count_symm_virt(unocc_symm_cts, occ_orbs, n_elec, symm);
            unsigned int n_virt = count_sing_virt(occ_orbs, n_elec, symm->symm_vec.data(), n_orb, unocc_symm_cts, &orb_indices2[samp_idx][1]);
            if (n_virt == 0) {
                ndiv[samp_idx] = 1;
                vec1[samp_idx] = 0;
            }
            else {
                ndiv[samp_idx] = n_virt;
                orb_indices2[samp_idx][3] = n_virt; // number of allowed virtual orbitals
            }
        }
    }
    if (proc_rank == 0) {
        rn_sys = mt_obj() / (1. + UINT32_MAX);
    }
    comp_len = comp_sub(vec1.data(), comp_len, ndiv.data(), subwts, keep_sub, new_hb ? nsub.data() : NULL, n_samp, wt_remain.data(), rn_sys, vec2.data(), comp_idx);
    if (comp_len > spawn_length) {
        std::cerr << "Error: insufficient memory allocated for matrix compression.\n";
    }
                
#pragma mark 1st unoccupied (double)
    subwts.reshape(spawn_length, n_orb - n_elec / 2);
    keep_sub.reshape(spawn_length, n_orb - n_elec / 2);
    for (size_t samp_idx = 0; samp_idx < comp_len; samp_idx++) {
        size_t weight_idx = comp_idx[samp_idx][0];
        size_t det_idx = det_indices1[weight_idx];
        det_indices2[samp_idx] = det_idx;
        orb_indices1[samp_idx][0] = orb_indices2[weight_idx][0]; // single or double
        uint8_t o1_idx = orb_indices2[weight_idx][1];
        orb_indices1[samp_idx][1] = o1_idx; // 1st occupied index
        uint8_t o2u1_orb = comp_idx[samp_idx][1]; // 2nd occupied orbital index (doubles), NOT converted to orbital below; unoccupied orbital index (singles)
        orb_indices1[samp_idx][2] = o2u1_orb;
        if (orb_indices1[samp_idx][0] == 0) { // double excitation
            if (o2u1_orb >= n_elec) {
                std::cerr << "Error: chosen occupied orbital (second) is out of bounds\n";
                vec2[samp_idx] = 0;
                ndiv[samp_idx] = 1;
                continue;
            }
            ndiv[samp_idx] = 0;
            uint8_t *occ_tmp = all_orbs[det_idx];
            //                orb_indices1[samp_idx][2] = occ_tmp[o2u1_orb];
            int o1_spin = o1_idx / (n_elec / 2);
            int o2_spin = occ_tmp[o2u1_orb] / n_orb;
            uint8_t o1_orb = occ_tmp[o1_idx];
            double tot_weight = calc_u1_probs(hb_probs, subwts[samp_idx], o1_orb, occ_tmp, n_elec, new_hb && (o1_spin == o2_spin));
            if (new_hb) {
                vec2[samp_idx] *= tot_weight;
            }
        }
        else { // single excitation
            uint8_t n_virt = orb_indices2[weight_idx][3];
            if (o2u1_orb >= n_virt) {
                vec1[samp_idx] = 0;
                std::cerr << "Error: index of chosen virtual orbital exceeds maximum\n";
            }
            orb_indices1[samp_idx][3] = n_virt;
            ndiv[samp_idx] = 1;
        }
    }
    if (proc_rank == 0) {
        rn_sys = mt_obj() / (1. + UINT32_MAX);
    }
    comp_len = comp_sub(vec2.data(), comp_len, ndiv.data(), subwts, keep_sub, NULL, n_samp, wt_remain.data(), rn_sys, vec1.data(), comp_idx);
    if (comp_len > spawn_length) {
        std::cerr << "Error: insufficient memory allocated for matrix compression.\n";
    }
                
#pragma mark 2nd unoccupied (double)
    subwts.reshape(spawn_length, symm->max_n_symm);
    keep_sub.reshape(spawn_length, symm->max_n_symm);
    for (size_t samp_idx = 0; samp_idx < comp_len; samp_idx++) {
        size_t weight_idx = comp_idx[samp_idx][0];
        size_t det_idx = det_indices2[weight_idx];
        det_indices1[samp_idx] = det_idx;
        orb_indices2[samp_idx][0] = orb_indices1[weight_idx][0]; // single or double
        uint8_t o1_idx = orb_indices1[weight_idx][1];
        orb_indices2[samp_idx][1] = o1_idx; // 1st occupied index
        uint8_t o2_idx = orb_indices1[weight_idx][2];
        orb_indices2[samp_idx][2] = o2_idx; // 2nd occupied index (doubles); unoccupied orbital index (singles)
        if (orb_indices2[samp_idx][0] == 0) { // double excitation
            uint8_t u1_orb = find_nth_virt(all_orbs[det_idx], o1_idx / (n_elec / 2), n_elec, n_orb, comp_idx[samp_idx][1]);
            uint8_t *curr_det = all_dets[det_idx];
            if (read_bit(curr_det, u1_orb)) { // now this really should never happen
                std::cerr << "Error: occupied orbital chosen as 1st virtual\n";
                vec1[samp_idx] = 0;
                ndiv[samp_idx] = 1;
            }
            else {
                ndiv[samp_idx] = 0;
                orb_indices2[samp_idx][3] = u1_orb;
                double tot_weight;
                uint8_t *occ_tmp = all_orbs[det_idx];
                uint8_t o1_orb = occ_tmp[o1_idx];
                uint8_t o2_orb = occ_tmp[o2_idx];
                if (new_hb) {
                    tot_weight = calc_u2_probs_half(hb_probs, subwts[samp_idx], o1_orb, o2_orb, u1_orb, curr_det, symm, &nsub[samp_idx]);
                }
                else {
                    tot_weight = calc_u2_probs(hb_probs, subwts[samp_idx], o1_orb, o2_orb, u1_orb, symm, &nsub[samp_idx]);
                }
                if (new_hb || tot_weight == 0) {
                    vec1[samp_idx] *= tot_weight;
                }
            }
        }
        else {
            orb_indices2[samp_idx][3] = orb_indices1[weight_idx][3];
            ndiv[samp_idx] = 1;
        }
    }
    if (proc_rank == 0) {
        rn_sys = mt_obj() / (1. + UINT32_MAX);
    }
    comp_len = comp_sub(vec1.data(), comp_len, ndiv.data(), subwts, keep_sub, nsub.data(), n_samp, wt_remain.data(), rn_sys, vec2.data(), comp_idx);
    if (comp_len > spawn_length) {
        std::cerr << "Error: insufficient memory allocated for matrix compression.\n";
    }
    comp_scratch->vec_len = comp_len;
}
