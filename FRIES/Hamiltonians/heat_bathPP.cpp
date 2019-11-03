/*! \file
 *
 * \brief Utilities for Heat-Bath Power-Pitzer compression of Hamiltonian
 *
 * These functions apply only to the double excitation portion of a molecular
 * Hamiltonian matrix. Single excitations are treated as in the near-uniform
 * scheme
 */

#include "heat_bathPP.hpp"

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
    
    double *diff_tab = tens->d_diff;
    unsigned int offset = (1 - o1_spin) * n_elec / 2;
    for (orb_idx = offset; orb_idx < (n_elec / 2 + offset); orb_idx++) {
        prob_arr[orb_idx] = diff_tab[(o1_orb % n_orb) * n_orb + occ_orbs[orb_idx] % n_orb];
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
                     const Matrix<unsigned char> &lookup_tabl, unsigned char *symm,
                     unsigned int *prob_len) {
    unsigned int n_orb = (unsigned int)tens->n_orb;
    unsigned char o2_spinless = o2_orb % n_orb;
    unsigned char u1_spinless = u1_orb % n_orb;
    int same_spin = (o1_orb / n_orb) == (o2_orb / n_orb);
    unsigned char u2_irrep = symm[o1_orb % n_orb] ^ symm[o2_spinless] ^ symm[u1_spinless];
    unsigned int num_u2 = lookup_tabl(u2_irrep, 0);
    if (*prob_len == 0) {
        *prob_len = num_u2;
    }
    unsigned int orb_idx;
    unsigned char u2_orb;
    unsigned char min_o2_u2, max_o2_u2;
    double norm = 0;
    for (orb_idx = 0; orb_idx < num_u2; orb_idx++) {
        u2_orb = lookup_tabl(u2_irrep, orb_idx + 1);
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
    for (orb_idx = num_u2; orb_idx < *prob_len; orb_idx++) {
        prob_arr[orb_idx] = 0;
    }
    if (norm != 0) {
        double inv_norm = 1 / norm;
        for (orb_idx = 0; orb_idx < num_u2; orb_idx++) {
            u2_orb = lookup_tabl(u2_irrep, orb_idx + 1);
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
        double *diff_tab = tens->d_diff;
        weight = (tens->s_tens[o1] * diff_tab[o1 * n_orb + o2] + tens->s_tens[o2] * diff_tab[o2 * n_orb + o1]) * tens->exch_sqrt[I_J_TO_TRI(min_o1_u1, max_o1_u1)] * tens->exch_sqrt[I_J_TO_TRI(min_o2_u2, max_o2_u2)];
    }
    return weight;
}


double calc_norm_wt(hb_info *tens, unsigned char *orbs, unsigned char *occ,
                    unsigned int n_elec, long long det,
                    const Matrix<unsigned char> &lookup_tabl, unsigned char *symm) {
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
    double *diff_tab = tens->d_diff;
    double d1_denom = 0;
    unsigned int offset = (1 - o1_spin) * n_elec / 2;
    for (orb_idx = offset; orb_idx < (n_elec / 2 + offset); orb_idx++) {
        d1_denom += diff_tab[o1 * n_orb + occ[orb_idx] % n_orb];;
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
        d2_denom += diff_tab[o2 * n_orb + occ[orb_idx] % n_orb];;
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
    
    unsigned char u1_irrep = symm[u1];
    unsigned char u2_irrep = symm[u2];
    unsigned char symm_orb, min_orb, max_orb;
    
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
        unsigned char min_o1_u2 = o1 < u2 ? o1 : u2;
        unsigned char max_o1_u2 = o1 > u2 ? o1 : u2;
        unsigned char min_o2_u1 = o2 < u1 ? o2 : u1;
        unsigned char max_o2_u1 = o2 > u1 ? o2 : u1;
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


unsigned int hb_doub_multi(long long det, unsigned char *occ_orbs,
                           unsigned int num_elec, unsigned char *orb_symm,
                           hb_info *tens, const Matrix<unsigned char> &lookup_tabl,
                           unsigned int num_sampl, mt_struct *rn_ptr,
                           unsigned char (* chosen_orbs)[4], double *prob_vec) {
    unsigned int num_orb = (unsigned int) tens->n_orb;
    unsigned int n_states = num_elec > num_orb ? num_elec : num_orb;
    unsigned int alias_idx[n_states];
    double alias_probs[n_states];
    double probs[n_states];
    
    // Choose first occupied orbital
    calc_o1_probs(tens, probs, num_elec, occ_orbs);
    setup_alias(probs, alias_idx, alias_probs, num_elec);
    sample_alias(alias_idx, alias_probs, num_elec, chosen_orbs[0], num_sampl, 4, rn_ptr);
    
    unsigned int o1_samples[num_elec];
    unsigned int samp_idx, elec_idx;
    for (elec_idx = 0; elec_idx < num_elec; elec_idx++) {
        o1_samples[elec_idx] = 0;
    }
    for (samp_idx = 0; samp_idx < num_sampl; samp_idx++) {
        o1_samples[chosen_orbs[samp_idx][0]]++;
    }
    
    unsigned int loc_n_samp;
    unsigned int tot_sampled = 0;
    for (elec_idx = 0; elec_idx < num_elec; elec_idx++) {
        loc_n_samp = o1_samples[elec_idx];
        if (loc_n_samp == 0) {
            continue;
        }
        unsigned char o1 = elec_idx;
        unsigned int samp_begin = tot_sampled;
        // Choose second occupied orbital
        calc_o2_probs(tens, probs, num_elec, occ_orbs, &o1);
        setup_alias(probs, alias_idx, alias_probs, num_elec);
        sample_alias(alias_idx, alias_probs, num_elec, &chosen_orbs[samp_begin][1], loc_n_samp, 4, rn_ptr);
        
        // Choose first virtual orbital
        calc_u1_probs(tens, probs, o1, det);
        setup_alias(probs, alias_idx, alias_probs, num_orb);
        sample_alias(alias_idx, alias_probs, num_orb, &chosen_orbs[samp_begin][2], loc_n_samp, 4, rn_ptr);
        
        for (samp_idx = samp_begin; samp_idx < samp_begin + loc_n_samp; samp_idx++) {
            unsigned char o2 = occ_orbs[chosen_orbs[samp_idx][1]];
            unsigned char u1 = chosen_orbs[samp_idx][2] + num_orb * (o1 / num_orb);
            unsigned char u2_symm = orb_symm[o1 % num_orb] ^ orb_symm[o2 % num_orb] ^ orb_symm[u1 % num_orb];
            unsigned int num_u2 = 0;
            double u2_norm = calc_u2_probs(tens, probs, o1, o2, u1, lookup_tabl, orb_symm, &num_u2);
            if (u2_norm != 0) {
                setup_alias(probs, alias_idx, alias_probs, num_u2);
                unsigned char u2;
                sample_alias(alias_idx, alias_probs, num_u2, &u2, 1, 1, rn_ptr);
                u2 = lookup_tabl(u2_symm, u2 + 1) + num_orb * (o2 / num_orb);
                if (det & (1LL << u2)) {
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
                prob_vec[tot_sampled] = calc_norm_wt(tens, &chosen_orbs[tot_sampled][0], occ_orbs, num_elec, det, lookup_tabl, orb_symm);
                tot_sampled++;
            }
        }
    }
    return tot_sampled;
}
