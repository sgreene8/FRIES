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
#include <FRIES/math_utils.h>

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
            s_tens[i] += d_same[I_J_TO_TRI_NODIAG(j, i)];
        }
        for (j = i + 1; j < n_orb; j++) {
            s_tens[i] += d_same[I_J_TO_TRI_NODIAG(i, j)];
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
            exch_norms[i] += exch_mat[I_J_TO_TRI_NODIAG(j, i)];
        }
        exch_norms[i] += diag_part[i];
        for (j = i + 1; j < n_orb; j++) {
            exch_norms[i] += exch_mat[I_J_TO_TRI_NODIAG(i, j)];
        }
    }
    hb_obj->exch_norms = exch_norms;
    return hb_obj;
}


hb_info *set_up(uint32_t tot_orb, uint32_t n_orb, const SymmERIs &eris) {
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
                        d_diff[i * n_orb + j] += fabs(eris.physicist(i + half_frz, j + half_frz, a, b)); // exchange terms are zero
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
                        d_same[tri_idx] += 2 * fabs(eris.physicist(i + half_frz, j + half_frz, a, b) - eris.physicist(i + half_frz, j + half_frz, b, a));
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
            s_tens[i] += d_same[I_J_TO_TRI_NODIAG(j, i)];
        }
        for (j = i + 1; j < n_orb; j++) {
            s_tens[i] += d_same[I_J_TO_TRI_NODIAG(i, j)];
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
            exch_mat[tri_idx] = sqrt(fabs(eris.physicist(i + half_frz, j + half_frz, j + half_frz, i + half_frz)));
            tri_idx++;
        }
    }
    hb_obj->exch_sqrt = exch_mat;
    
    double *diag_part = (double *)malloc(sizeof(double) * n_orb);
    for (j = 0; j < n_orb; j++) {
        diag_part[j] = sqrt(fabs(eris.physicist(j + half_frz, j + half_frz, j + half_frz, j + half_frz)));
    }
    hb_obj->diag_sqrt = diag_part;
    
    double *exch_norms = (double *)calloc(n_orb, sizeof(double));
    for (i = 0; i < n_orb; i++) {
        for (j = 0; j < i; j++) {
            exch_norms[i] += exch_mat[I_J_TO_TRI_NODIAG(j, i)];
        }
        exch_norms[i] += diag_part[i];
        for (j = i + 1; j < n_orb; j++) {
            exch_norms[i] += exch_mat[I_J_TO_TRI_NODIAG(i, j)];
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
        prob_arr[orb_idx] = tens->d_same[I_J_TO_TRI_NODIAG(occ_orbs[orb_idx] % n_orb, o1_orb % n_orb)];
        norm += prob_arr[orb_idx];
    }
    for (unsigned int orb_idx = o1_idx + 1; orb_idx < (n_elec / 2 + offset); orb_idx++) {
        prob_arr[orb_idx] = tens->d_same[I_J_TO_TRI_NODIAG(o1_orb % n_orb, occ_orbs[orb_idx] % n_orb)];
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
            prob_arr[orb_idx] = tens->d_same[I_J_TO_TRI_NODIAG(occ_orbs[orb_idx], o1_orb)];
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
            prob_arr[orb_idx] = tens->d_same[I_J_TO_TRI_NODIAG(occ_orbs[orb_idx] - n_orb, o1_orb - n_orb)];
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
            prob_arr[prob_idx] = tens->exch_sqrt[I_J_TO_TRI_NODIAG(orb_idx, o1_spinless)];
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
            prob_arr[prob_idx] = tens->exch_sqrt[I_J_TO_TRI_NODIAG(o1_spinless, orb_idx)];
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
                prob_arr[orb_idx] = tens->exch_sqrt[I_J_TO_TRI_NODIAG(min_o2_u2, max_o2_u2)];
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
                prob_arr[orb_idx] = tens->exch_sqrt[I_J_TO_TRI_NODIAG(min_o2_u2, max_o2_u2)];
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
        uint32_t o1o2 = I_J_TO_TRI_NODIAG(o1, o2);
        uint32_t o1u1 = I_J_TO_TRI_NODIAG(min_o1_u1, max_o1_u1);
        uint32_t o2u2 = I_J_TO_TRI_NODIAG(min_o2_u2, max_o2_u2);
        weight = tens->d_same[o1o2] * (tens->exch_sqrt[o1u1] * tens->exch_sqrt[o2u2]) / tens->s_norm / tens->exch_norms[o1] / tens->exch_norms[o2];
    }
    else {
        double *diff_tab = tens->d_diff;
        uint32_t o1u1idx = I_J_TO_TRI_NODIAG(min_o1_u1, max_o1_u1);
        uint32_t o2u2idx = I_J_TO_TRI_NODIAG(min_o2_u2, max_o2_u2);
        weight = (diff_tab[o2 * n_orb + o1]) * tens->exch_sqrt[o1u1idx] * tens->exch_sqrt[o2u2idx] / tens->s_norm / tens->exch_norms[o1] / tens->exch_norms[o2];
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
        d1_denom += tens->d_same[I_J_TO_TRI_NODIAG(occ_spatial[orb_idx], o1)];
    }
    for (orb_idx++; orb_idx < (n_elec / 2 + offset); orb_idx++) {
        d1_denom += tens->d_same[I_J_TO_TRI_NODIAG(o1, occ_spatial[orb_idx])];
    }
    double d2_denom = 0;
    offset = (1 - o2_spin) * n_elec / 2;
    for (orb_idx = offset; orb_idx < (n_elec / 2 + offset); orb_idx++) {
        d2_denom += diff_tab[o2 * n_orb + occ_spatial[orb_idx]];;
    }
    offset = o2_spin * n_elec / 2;
    for (orb_idx = offset; occ_spatial[orb_idx] < o2; orb_idx++) {
        d2_denom += tens->d_same[I_J_TO_TRI_NODIAG(occ_spatial[orb_idx], o2)];
    }
    for (orb_idx++; orb_idx < (n_elec / 2 + offset); orb_idx++) {
        d2_denom += tens->d_same[I_J_TO_TRI_NODIAG(o2, occ_spatial[orb_idx])];
    }
    
    double e1_virt = 0;
    offset = o1_spin * n_orb;
    for (orb_idx = 0; orb_idx < o1; orb_idx++) {
        if (!read_bit(det, orb_idx + offset)) {
            e1_virt += tens->exch_sqrt[I_J_TO_TRI_NODIAG(orb_idx, o1)];
        }
    }
    for (orb_idx = o1 + 1; orb_idx < n_orb; orb_idx++) {
        if (!read_bit(det, orb_idx + offset)) {
            e1_virt += tens->exch_sqrt[I_J_TO_TRI_NODIAG(o1, orb_idx)];
        }
    }
    
    double e2_virt = 0;
    offset = o2_spin * n_orb;
    for (orb_idx = 0; orb_idx < o2; orb_idx++) {
        if (!read_bit(det, orb_idx + offset)) {
            e2_virt += tens->exch_sqrt[I_J_TO_TRI_NODIAG(orb_idx, o2)];
        }
    }
    for (orb_idx = o2 + 1; orb_idx < n_orb; orb_idx++) {
        if (!read_bit(det, orb_idx + offset)) {
            e2_virt += tens->exch_sqrt[I_J_TO_TRI_NODIAG(o2, orb_idx)];
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
                e2_symm_no1 += tens->exch_sqrt[I_J_TO_TRI_NODIAG(min_orb, max_orb)];
            }
        }
        if ((same_sp && symm_orb != u1) || !same_sp) {
            if (o1 == symm_orb) {
                e1_symm_no1 += tens->diag_sqrt[o1];
            }
            else {
                min_orb = (o1 < symm_orb) ? o1 : symm_orb;
                max_orb = (o1 > symm_orb) ? o1 : symm_orb;
                e1_symm_no1 += tens->exch_sqrt[I_J_TO_TRI_NODIAG(min_orb, max_orb)];
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
                e2_symm_no2 += tens->exch_sqrt[I_J_TO_TRI_NODIAG(min_orb, max_orb)];
            }
        }
        if ((same_sp && symm_orb != u2) || !same_sp) {
            if (o1 == symm_orb) {
                e1_symm_no2 += tens->diag_sqrt[o1];
            }
            else {
                min_orb = (o1 < symm_orb) ? o1 : symm_orb;
                max_orb = (o1 > symm_orb) ? o1 : symm_orb;
                e1_symm_no2 += tens->exch_sqrt[I_J_TO_TRI_NODIAG(min_orb, max_orb)];
            }
        }
    }
    
    unsigned int o1u1tri = I_J_TO_TRI_NODIAG(min_o1_u1, max_o1_u1);
    unsigned int o2u2tri = I_J_TO_TRI_NODIAG(min_o2_u2, max_o2_u2);
    if (same_sp) {
        uint8_t min_o1_u2 = o1 < u2 ? o1 : u2;
        uint8_t max_o1_u2 = o1 > u2 ? o1 : u2;
        uint8_t min_o2_u1 = o2 < u1 ? o2 : u1;
        uint8_t max_o2_u1 = o2 > u1 ? o2 : u1;
        unsigned int o1o2tri = I_J_TO_TRI_NODIAG(o1, o2);
        unsigned int o1u2tri = I_J_TO_TRI_NODIAG(min_o1_u2, max_o1_u2);
        unsigned int o2u1tri = I_J_TO_TRI_NODIAG(min_o2_u1, max_o2_u1);
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


void apply_HBPP_sys(Matrix<uint8_t> &all_orbs, Matrix<uint8_t> &all_dets, HBCompressSys *comp_scratch,
                    hb_info *hb_probs, SymmInfo *symm, double p_doub, bool new_hb,
                    std::mt19937 &mt_obj, uint32_t n_samp,
                    std::function<double(uint8_t *, uint8_t *)> sing_mat_fxn,
                    std::function<double(uint8_t *)> doub_mat_fxn) {
    std::vector<double> &vec1 = comp_scratch->vec1;
    std::vector<double> &vec2 = comp_scratch->vec2;
    Matrix<double> &subwts = comp_scratch->subwts;
    Matrix<bool> &keep_sub = comp_scratch->keep_sub;
    std::vector<uint32_t> &ndiv = comp_scratch->ndiv;
    std::vector<uint16_t> &nsub = comp_scratch->group_sizes;
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
            uint32_t n_occ = count_sing_allowed(occ_orbs, n_elec, symm->symm_vec.data(), n_orb, unocc_symm_cts);
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
            uint32_t n_virt = count_sing_virt(occ_orbs, n_elec, symm->symm_vec.data(), n_orb, unocc_symm_cts, &orb_indices2[samp_idx][1]);
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
    
    size_t num_success = 0;
    for (size_t samp_idx = 0; samp_idx < comp_len; samp_idx++) {
        size_t weight_idx = comp_idx[samp_idx][0];
        size_t det_idx = det_indices1[weight_idx];
        det_indices2[num_success] = det_idx;
        uint8_t *occ_orbs = all_orbs[det_idx];
        uint8_t *curr_det = all_dets[det_idx];
        uint8_t o1_idx = orb_indices2[weight_idx][1];
        if (orb_indices2[weight_idx][0] == 0) { // double excitation
            uint8_t o2_idx = orb_indices2[weight_idx][2];
            uint8_t o1_orb = occ_orbs[o1_idx];
            uint8_t o2_orb = occ_orbs[o2_idx];
            uint8_t u1_orb = orb_indices2[weight_idx][3];
            uint8_t u2_symm = symm->symm_vec[o1_orb % n_orb] ^ symm->symm_vec[o2_orb % n_orb] ^ symm->symm_vec[u1_orb % n_orb];
            uint8_t u2_orb = symm->symm_lookup[u2_symm][comp_idx[samp_idx][1] + 1] + n_orb * (o2_orb / n_orb);
            if (read_bit(curr_det, u2_orb)) { // chosen orbital is occupied; unsuccessful spawn
                if (new_hb) {
                    std::cerr << "Error: occupied orbital chosen as second virtual in unnormalized heat-bath\n";
                }
                continue;
            }
            if (u1_orb == u2_orb) { // This shouldn't happen, but in case it does (e.g. by numerical error)
                std::cerr << "Error: repeat virtual orbital chosen\n";
                continue;
            }
            if (u1_orb > u2_orb) {
                std::swap(u1_orb, u2_orb);
            }
            if (o1_orb > o2_orb) {
                std::swap(o1_orb, o2_orb);
            }
            orb_indices1[num_success][0] = o1_orb;
            orb_indices1[num_success][1] = o2_orb;
            orb_indices1[num_success][2] = u1_orb;
            orb_indices1[num_success][3] = u2_orb;
            double tot_weight;
            if (new_hb) {
                tot_weight = calc_unnorm_wt(hb_probs, orb_indices1[num_success]);
            }
            else {
                tot_weight = calc_norm_wt(hb_probs, orb_indices1[num_success], occ_orbs, n_elec, curr_det, symm);
            }
            double vec_el = doub_mat_fxn(orb_indices1[num_success]) * vec2[samp_idx] / tot_weight / p_doub;
            if (fabs(vec_el) > 1e-9) {
                vec_el *= doub_parity(curr_det, orb_indices1[num_success]);
                vec1[num_success] = vec_el;
                num_success++;
            }
        }
        else {
            uint8_t o1_orb = occ_orbs[o1_idx];
            orb_indices1[num_success][0] = o1_orb;
            uint8_t u1_symm = symm->symm_vec[o1_orb % n_orb];
            uint8_t spin = o1_orb / n_orb;
            uint8_t u1_orb = virt_from_idx(curr_det, symm->symm_lookup[u1_symm], n_orb * spin, orb_indices2[weight_idx][2]);
            if (u1_orb == 255) {
                std::cerr << "Error: virtual orbital not found\n";
                continue;
            }
            orb_indices1[num_success][1] = u1_orb;
            orb_indices1[num_success][2] = orb_indices1[num_success][3] = 0;
            count_symm_virt(unocc_symm_cts, occ_orbs, n_elec, symm);
            unsigned int n_occ = count_sing_allowed(occ_orbs, n_elec, symm->symm_vec.data(), n_orb, unocc_symm_cts);
            
            double vec_el = sing_mat_fxn(orb_indices1[num_success], occ_orbs);
            vec_el *= vec2[samp_idx] / (1 - p_doub) * n_occ * orb_indices2[weight_idx][3];
            if (fabs(vec_el) > 1e-9) {
                vec_el *= sing_parity(curr_det, orb_indices1[num_success]);
                vec1[num_success] = vec_el;
                num_success++;
            }
        }
    }
    
    comp_scratch->vec_len = num_success;
}

size_t collapse_long_(std::vector<double> &short_vec, std::vector<double> &long_vec,
                      size_t short_len, size_t long_len, std::vector<bool> &keep_vec,
                      std::vector<uint16_t> &group_sizes,
                      std::function<void(size_t, size_t, size_t)> transfer_fxn) {
    size_t n_short = 0;
    size_t long_idx = 0;
    for (size_t short_idx = 0; short_idx < short_len; short_idx++) {
        for (size_t group_idx = 0; group_idx < group_sizes[short_idx]; group_idx++) {
            if (!keep_vec[long_idx]) { // element was not zeroed in compression
                short_vec[n_short] = long_vec[long_idx];
                transfer_fxn(short_idx, n_short, group_idx);
                n_short++;
            }
            keep_vec[long_idx] = false;
            long_idx++;
        }
    }
    return n_short;
}

void apply_HBPP_piv(Matrix<uint8_t> &all_orbs, Matrix<uint8_t> &all_dets, HBCompressPiv *comp_scratch,
                    hb_info *hb_probs, SymmInfo *symm, double p_doub, bool new_hb,
                    std::mt19937 &mt_obj, uint32_t n_samp,
                    std::function<double(uint8_t *, uint8_t *)> sing_mat_fxn,
                    std::function<double(uint8_t *)> doub_mat_fxn, int spin_parity) {
    if (spin_parity && !new_hb) {
        throw std::runtime_error("Time-reversal symmetry is only supported for the modified HB-PP distribution.");
    }
    
    std::vector<double> &short_vec = comp_scratch->vec1;
    std::vector<double> &long_vec = comp_scratch->long_vec;
    size_t n_short = comp_scratch->vec_len;
    std::vector<size_t> &det_indices1 = comp_scratch->det_indices1;
    std::vector<size_t> &det_indices2 = comp_scratch->det_indices2;
    uint8_t (*orb_indices1)[4] = comp_scratch->orb_indices1;
    uint8_t (*orb_indices2)[4] = comp_scratch->orb_indices2;
    std::vector<size_t> &srt_arr = comp_scratch->cmp_srt;
    std::vector<bool> &keep_arr = comp_scratch->keep_idx;
    std::vector<uint16_t> &group_sizes = comp_scratch->group_sizes;
    
    int proc_rank = 0;
    int n_procs = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    uint32_t n_elec = (uint32_t) all_orbs.cols();
    uint32_t n_orb = (uint32_t) hb_probs->n_orb;
    size_t det_size = CEILING(2 * n_orb, 8);
    unsigned int unocc_symm_cts[n_irreps][2];
    
    std::function<void(size_t, size_t, size_t)> transfer_fxn;
    
#pragma mark Singles vs doubles
    size_t n_long = 0;
    for (size_t short_idx = 0; short_idx < n_short; short_idx++) {
        double weight = fabs(short_vec[short_idx]);
        if (weight > 0) {
            long_vec[n_long] = weight * p_doub;
            n_long++;
            long_vec[n_long] = weight * (1 - p_doub);
            n_long++;
            group_sizes[short_idx] = 2;
        }
        else {
            group_sizes[short_idx] = 0;
        }
    }
    piv_comp_parallel(long_vec.data(), n_long, n_samp, srt_arr, keep_arr, mt_obj);
    transfer_fxn = [&det_indices1, &det_indices2, orb_indices1](size_t old_short, size_t new_short, size_t group) {
        det_indices2[new_short] = det_indices1[old_short];
        orb_indices1[new_short][0] = group;
    };
    n_short = collapse_long_(short_vec, long_vec, n_short, n_long, keep_arr, group_sizes, transfer_fxn);

#pragma mark  First occupied orbital
    n_long = 0;
    for (size_t short_idx = 0; short_idx < n_short; short_idx++) {
        size_t det_idx = det_indices2[short_idx];
        uint8_t *occ_orbs = all_orbs[det_idx];
        if (orb_indices1[short_idx][0] == 0) { // double excitation
            group_sizes[short_idx] = n_elec - new_hb;
            double tot_weight = calc_o1_probs(hb_probs, &long_vec[n_long], n_elec, occ_orbs, new_hb);
            for (size_t prob_idx = 0; prob_idx < n_elec - new_hb; prob_idx++) {
                long_vec[n_long + prob_idx] *= short_vec[short_idx] * (new_hb ? tot_weight : 1);
            }
            n_long += n_elec - new_hb;
        }
        else { // single excitation
            count_symm_virt(unocc_symm_cts, occ_orbs, n_elec, symm);
            uint32_t n_occ = count_sing_allowed(occ_orbs, n_elec, symm->symm_vec.data(), n_orb, unocc_symm_cts);
            group_sizes[short_idx] = n_occ;
            if (n_occ != 0) {
                for (size_t orb_idx = 0; orb_idx < n_occ; orb_idx++) {
                    long_vec[n_long + orb_idx] = short_vec[short_idx] / n_occ;
                }
                n_long += n_occ;
            }
        }
    }
    piv_comp_parallel(long_vec.data(), n_long, n_samp, srt_arr, keep_arr, mt_obj);
    transfer_fxn = [&det_indices1, &det_indices2, orb_indices1, orb_indices2](size_t old_short, size_t new_short, size_t group) {
        det_indices1[new_short] = det_indices2[old_short];
        orb_indices2[new_short][0] = orb_indices1[old_short][0]; // single or double
        orb_indices2[new_short][1] = group; // first occupied orbital index (NOT converted to orbital below)
    };
    n_short = collapse_long_(short_vec, long_vec, n_short, n_long, keep_arr, group_sizes, transfer_fxn);

#pragma mark Unoccupied orbital (single); 2nd occupied (double)
    n_long = 0;
    for (size_t short_idx = 0; short_idx < n_short; short_idx++) {
        size_t det_idx = det_indices1[short_idx];
        if (orb_indices2[short_idx][1] >= n_elec) {
            std::cerr << "Error: chosen occupied orbital (first) is out of bounds\n";
            group_sizes[short_idx] = 0;
            continue;
        }
        uint8_t *occ_orbs = all_orbs[det_idx];
        if (orb_indices2[short_idx][0] == 0) { // double excitation
            double tot_weight = 1;
            uint16_t n_o2;
            if (new_hb) {
                orb_indices2[short_idx][1]++;
                n_o2 = orb_indices2[short_idx][1];
                tot_weight = calc_o2_probs_half(hb_probs, &long_vec[n_long], n_elec, occ_orbs, n_o2);
            }
            else {
                n_o2 = n_elec;
                calc_o2_probs(hb_probs, &long_vec[n_long], n_elec, occ_orbs, orb_indices2[short_idx][1]);
            }
            for (size_t prob_idx = 0; prob_idx < n_o2; prob_idx++) {
                long_vec[n_long + prob_idx] *= tot_weight * short_vec[short_idx];
            }
            group_sizes[short_idx] = n_o2;
            n_long += n_o2;
        }
        else { // single excitation
            count_symm_virt(unocc_symm_cts, occ_orbs, n_elec, symm);
            uint32_t n_virt = count_sing_virt(occ_orbs, n_elec, symm->symm_vec.data(), n_orb, unocc_symm_cts, &orb_indices2[short_idx][1]);
            if (n_virt == 0) {
                group_sizes[short_idx] = 0;
            }
            else {
                group_sizes[short_idx] = n_virt;
                orb_indices2[short_idx][3] = n_virt; // number of allowed virtual orbitals
                for (size_t prob_idx = 0; prob_idx < n_virt; prob_idx++) {
                    long_vec[n_long + prob_idx] = short_vec[short_idx] / n_virt;
                }
                n_long += n_virt;
            }
        }
    }
    piv_comp_parallel(long_vec.data(), n_long, n_samp, srt_arr, keep_arr, mt_obj);
    transfer_fxn = [&det_indices1, &det_indices2, orb_indices1, orb_indices2](size_t old_short, size_t new_short, size_t group) {
        bool single = orb_indices2[old_short][0] == 1;
        det_indices2[new_short] = det_indices1[old_short];
        orb_indices1[new_short][0] = orb_indices2[old_short][0]; // single or double
        orb_indices1[new_short][1] = orb_indices2[old_short][1]; // first occupied orbital index
        orb_indices1[new_short][2] = group; // 2nd occupied orbital index (doubles), NOT converted to orbital below; unoccupied orbital index (singles)
        if (single) {
            orb_indices1[new_short][3] = orb_indices2[old_short][3];
        }
    };
    n_short = collapse_long_(short_vec, long_vec, n_short, n_long, keep_arr, group_sizes, transfer_fxn);

    #pragma mark 1st unoccupied (double)
    n_long = 0;
    for (size_t short_idx = 0; short_idx < n_short; short_idx++) {
        size_t det_idx = det_indices2[short_idx];
        uint8_t o2u1_orb = orb_indices1[short_idx][2];
        if (orb_indices1[short_idx][0] == 0) { // double excitation
            if (o2u1_orb >= n_elec) {
                std::cerr << "Error: chosen occupied orbital (second) is out of bounds\n";
                group_sizes[short_idx] = 0;
                continue;
            }
            uint8_t *occ_tmp = all_orbs[det_idx];
            uint8_t o1_idx = orb_indices1[short_idx][1];
            int o1_spin = o1_idx / (n_elec / 2);
            int o2_spin = occ_tmp[o2u1_orb] / n_orb;
            uint8_t o1_orb = occ_tmp[o1_idx];
            double tot_weight = calc_u1_probs(hb_probs, &long_vec[n_long], o1_orb, occ_tmp, n_elec, new_hb && (o1_spin == o2_spin));
            uint32_t n_virt = n_orb - n_elec / 2;
            group_sizes[short_idx] = n_virt;
            for (size_t prob_idx = 0; prob_idx < n_virt; prob_idx++) {
                long_vec[n_long + prob_idx] *= short_vec[short_idx] * (new_hb ? tot_weight : 1);
            }
            n_long += n_virt;
        }
        else { // single excitation
            uint8_t n_virt = orb_indices1[short_idx][3];
            if (o2u1_orb >= n_virt) {
                group_sizes[short_idx] = 0;
                std::cerr << "Error: index of chosen virtual orbital exceeds maximum\n";
            }
            group_sizes[short_idx] = 1;
            long_vec[n_long] = short_vec[short_idx];
            n_long++;
        }
    }
    piv_comp_parallel(long_vec.data(), n_long, n_samp, srt_arr, keep_arr, mt_obj);
    transfer_fxn = [&det_indices1, &det_indices2, orb_indices1, orb_indices2](size_t old_short, size_t new_short, size_t group) {
        bool single = orb_indices1[old_short][0] == 1;
        det_indices1[new_short] = det_indices2[old_short];
        orb_indices2[new_short][0] = orb_indices1[old_short][0]; // single or double
        orb_indices2[new_short][1] = orb_indices1[old_short][1]; // first occupied orbital index
        orb_indices2[new_short][2] = orb_indices1[old_short][2]; // 2nd occupied index (doubles); unoccupied orbital index (singles)
        if (single) {
            orb_indices2[new_short][3] = orb_indices1[old_short][3];
        }
        else {
            orb_indices2[new_short][3] = group;
        }
    };
    n_short = collapse_long_(short_vec, long_vec, n_short, n_long, keep_arr, group_sizes, transfer_fxn);

#pragma mark 2nd unoccupied (double)
    n_long = 0;
    for (size_t short_idx = 0; short_idx < n_short; short_idx++) {
        size_t det_idx = det_indices1[short_idx];
        uint8_t o1_idx = orb_indices2[short_idx][1];
        if (orb_indices2[short_idx][0] == 0) { // double excitation
            uint8_t u1_orb = find_nth_virt(all_orbs[det_idx], o1_idx / (n_elec / 2), n_elec, n_orb, orb_indices2[short_idx][3]);
            uint8_t *curr_det = all_dets[det_idx];
            if (read_bit(curr_det, u1_orb)) { // now this really should never happen
                std::cerr << "Error: occupied orbital chosen as 1st virtual\n";
                group_sizes[short_idx] = 0;
            }
            else {
                orb_indices2[short_idx][3] = u1_orb;
                double tot_weight;
                uint8_t *occ_tmp = all_orbs[det_idx];
                uint8_t o1_orb = occ_tmp[o1_idx];
                uint8_t o2_orb = occ_tmp[orb_indices2[short_idx][2]];
                uint16_t n_probs;
                if (new_hb) {
                    tot_weight = calc_u2_probs_half(hb_probs, &long_vec[n_long], o1_orb, o2_orb, u1_orb, curr_det, symm, &n_probs);
                }
                else {
                    tot_weight = calc_u2_probs(hb_probs, &long_vec[n_long], o1_orb, o2_orb, u1_orb, symm, &n_probs);
                }
                for (size_t prob_idx = 0; prob_idx < n_probs; prob_idx++) {
                    long_vec[n_long + prob_idx] *= short_vec[short_idx] * (new_hb ? tot_weight : 1);
                }
                group_sizes[short_idx] = n_probs;
                n_long += n_probs;
            }
        }
        else { // single excitation
            group_sizes[short_idx] = 1;
            long_vec[n_long] = short_vec[short_idx];
            n_long++;
        }
    }
    piv_comp_parallel(long_vec.data(), n_long, n_samp, srt_arr, keep_arr, mt_obj);
    
    size_t old_short_len = n_short;
    n_short = 0;
    size_t long_idx = 0;
    for (size_t short_idx = 0; short_idx < old_short_len; short_idx++) {
        for (size_t group_idx = 0; group_idx < group_sizes[short_idx]; group_idx++) {
            if (!keep_arr[long_idx]) {
                short_vec[n_short] = long_vec[long_idx];
                size_t det_idx = det_indices1[short_idx];
                det_indices2[n_short] = det_idx;
                uint8_t *occ_orbs = all_orbs[det_idx];
                uint8_t *curr_det = all_dets[det_idx];
                uint8_t o1_idx = orb_indices2[short_idx][1];
                double tot_weight;
                uint8_t new_det[det_size];
                double matr_el;
                std::copy(curr_det, curr_det + det_size, new_det);
                if (orb_indices2[short_idx][0] == 0) { // double excitation
                    uint8_t o2_idx = orb_indices2[short_idx][2];
                    uint8_t o1_orb = occ_orbs[o1_idx];
                    uint8_t o2_orb = occ_orbs[o2_idx];
                    uint8_t u1_orb = orb_indices2[short_idx][3];
                    uint8_t u2_symm = symm->symm_vec[o1_orb % n_orb] ^ symm->symm_vec[o2_orb % n_orb] ^ symm->symm_vec[u1_orb % n_orb];
                    uint8_t u2_orb = symm->symm_lookup[u2_symm][group_idx + 1] + n_orb * (o2_orb / n_orb);
                    if (read_bit(curr_det, u2_orb)) { // chosen orbital is occupied; unsuccessful spawn
                        if (new_hb) {
                            std::cerr << "Error: occupied orbital chosen as second virtual in unnormalized heat-bath\n";
                        }
                        keep_arr[long_idx] = false;
                        long_idx++;
                        continue;
                    }
                    if (u1_orb == u2_orb) { // This shouldn't happen, but in case it does (e.g. by numerical error)
                        std::cerr << "Error: repeat virtual orbital chosen\n";
                        keep_arr[long_idx] = false;
                        long_idx++;
                        continue;
                    }
                    if (u1_orb > u2_orb) {
                        std::swap(u1_orb, u2_orb);
                    }
                    if (o1_orb > o2_orb) {
                        std::swap(o1_orb, o2_orb);
                    }
                    orb_indices1[n_short][0] = o1_orb;
                    orb_indices1[n_short][1] = o2_orb;
                    orb_indices1[n_short][2] = u1_orb;
                    orb_indices1[n_short][3] = u2_orb;
                    if (new_hb) {
                        tot_weight = calc_unnorm_wt(hb_probs, orb_indices1[n_short]);
                    }
                    else {
                        tot_weight = calc_norm_wt(hb_probs, orb_indices1[n_short], occ_orbs, n_elec, curr_det, symm);
                    }
                    tot_weight *= p_doub;
                    matr_el = doub_mat_fxn(orb_indices1[n_short]);
                    matr_el *= doub_det_parity(new_det, orb_indices1[n_short]);
                }
                else {
                    uint8_t o1_orb = occ_orbs[o1_idx];
                    orb_indices1[n_short][0] = o1_orb;
                    uint8_t u1_symm = symm->symm_vec[o1_orb % n_orb];
                    uint8_t spin = o1_orb / n_orb;
                    uint8_t u1_orb = virt_from_idx(curr_det, symm->symm_lookup[u1_symm], n_orb * spin, orb_indices2[short_idx][2]);
                    if (u1_orb == 255) {
                        std::cerr << "Error: virtual orbital not found\n";
                        keep_arr[long_idx] = false;
                        long_idx++;
                        continue;
                    }
                    orb_indices1[n_short][1] = u1_orb;
                    orb_indices1[n_short][2] = orb_indices1[n_short][3] = 0;
                    count_symm_virt(unocc_symm_cts, occ_orbs, n_elec, symm);
                    unsigned int n_occ = count_sing_allowed(occ_orbs, n_elec, symm->symm_vec.data(), n_orb, unocc_symm_cts);
                    tot_weight = (1 - p_doub) / n_occ / orb_indices2[short_idx][3];
                    
                    matr_el = sing_mat_fxn(orb_indices1[n_short], occ_orbs);
                    matr_el *= sing_det_parity(new_det, orb_indices1[n_short]);
                }
                if (spin_parity) {
                    double norm_factor;
                    uint8_t flipped_det[det_size];
                    flip_spins(curr_det, flipped_det, n_orb);
                    if (memcmp(curr_det, flipped_det, det_size) == 0) {
                        norm_factor = sqrt(2);
                    }
                    else {
                        norm_factor = 1;
                    }
                    flip_spins(new_det, flipped_det, n_orb);
                    if (memcmp(flipped_det, curr_det, det_size) == 0) {
                        keep_arr[long_idx] = false;
                        long_idx++;
                        continue; // choosing to exclude this b/c it's included in the diagonal
                    }
                    int cmp = memcmp(new_det, flipped_det, det_size);
                    if (cmp == 0) {
                        if (spin_parity == -1) { // matrix element is 0
                            keep_arr[long_idx] = false;
                            long_idx++;
                            continue;
                        }
                        matr_el *= 2;
                        norm_factor *= sqrt(2);
                    }
                    else {
                        uint8_t diff_orbs[4];
                        uint8_t n_bits_diff = find_diff_bits(curr_det, flipped_det, diff_orbs, det_size);
                        std::vector<uint8_t> &sv = symm->symm_vec;
                        if (n_bits_diff == 2) { // single excitation
                            if (sv[diff_orbs[0] % n_orb] == sv[diff_orbs[1] % n_orb]) {
                                if (read_bit(curr_det, diff_orbs[1])) {
                                    std::swap(diff_orbs[0], diff_orbs[1]);
                                }
                                count_symm_virt(unocc_symm_cts, occ_orbs, n_elec, symm);
                                uint32_t n_occ = count_sing_allowed(occ_orbs, n_elec, symm->symm_vec.data(), n_orb, unocc_symm_cts);
                                uint32_t n_virt = unocc_symm_cts[symm->symm_vec[diff_orbs[0] % n_orb]][0];
                                tot_weight += (1 - p_doub) / n_occ / n_virt;
                                
                                double rev_matr_el = sing_mat_fxn(diff_orbs, occ_orbs);
                                rev_matr_el *= sing_parity(curr_det, diff_orbs);
                                matr_el += rev_matr_el * spin_parity;
                            }
                        }
                        else if (n_bits_diff == 4) { // double excitation
                            if (sv[diff_orbs[0] % n_orb] ^ sv[diff_orbs[1] % n_orb] ^ sv[diff_orbs[2] % n_orb] ^ sv[diff_orbs[3] % n_orb] == 0) {
                                if (read_bit(curr_det, diff_orbs[2])) {
                                    if (read_bit(curr_det, diff_orbs[0])) {
                                        std::swap(diff_orbs[1], diff_orbs[2]);
                                    }
                                    else {
                                        std::swap(diff_orbs[0], diff_orbs[2]);
                                    }
                                }
                                if (read_bit(curr_det, diff_orbs[3])) {
                                    if (read_bit(curr_det, diff_orbs[0])) {
                                        std::swap(diff_orbs[1], diff_orbs[3]);
                                    }
                                    else {
                                        std::swap(diff_orbs[0], diff_orbs[3]);
                                    }
                                }
                                if (diff_orbs[0] > diff_orbs[1]) {
                                    std::swap(diff_orbs[0], diff_orbs[1]);
                                }
                                if (diff_orbs[2] > diff_orbs[3]) {
                                    std::swap(diff_orbs[2], diff_orbs[3]);
                                }
                                tot_weight += calc_unnorm_wt(hb_probs, diff_orbs) * p_doub;
                                
                                double rev_matr_el = doub_mat_fxn(diff_orbs);
                                rev_matr_el *= doub_parity(curr_det, diff_orbs);
                                matr_el += rev_matr_el * spin_parity;
                            }
                        }
                    }
                    if (cmp > 0) {
                        norm_factor *= spin_parity;
                    }
                    matr_el /= norm_factor;
                }
                double vec_el = long_vec[long_idx] * matr_el / tot_weight;
                if (fabs(vec_el) > 1e-12) {
                    short_vec[n_short] = vec_el;
                    n_short++;
                }
            }
            keep_arr[long_idx] = false;
            long_idx++;
        }
    }
    comp_scratch->vec_len = n_short;
}
