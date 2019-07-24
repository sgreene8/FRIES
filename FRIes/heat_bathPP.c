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
                    if (a != j && a != i && b != j && b != i) {
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
    
    return hb_obj;
}
