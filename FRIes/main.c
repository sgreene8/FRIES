//
//  main.c
//  FRIes
//
//  Created by Samuel Greene on 3/30/19.
//  Copyright © 2019 Samuel Greene. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "io_utils.h"
#include "fci_utils.h"
#include "near_uniform.h"
#include "dc.h"
#include "compress_utils.h"
#include "det_store.h"
#define max_n_dets 300000
#define max_iter 800000
#define target_walkers 250000


void adjust_shift(double *shift, unsigned int n_walk, unsigned int *last_nwalk,
                  unsigned int target_nwalk, double damp_factor) {
    if (*last_nwalk) {
        *shift -= damp_factor * log((double)n_walk / *last_nwalk);
        *last_nwalk = n_walk;
    }
    if (*last_nwalk == 0 && n_walk > target_nwalk) {
        *last_nwalk = n_walk;
    }
}

double calc_est_num(long long *vec_dets, int *vec_vals, long long *hf_dets,
                    double *hf_mel, size_t num_hf, hash_table *vec_hash,
                    unsigned long long *hf_hashes) {
    size_t hf_idx;
    ssize_t *ht_ptr;
    double numer = 0;
    for (hf_idx = 0; hf_idx < num_hf; hf_idx++) {
        ht_ptr = read_ht(vec_hash, hf_dets[hf_idx], hf_hashes[hf_idx], 0);
        if (ht_ptr) {
            numer += hf_mel[hf_idx] * vec_vals[*ht_ptr];
        }
    }
    return numer;
}


int main(int argc, const char * argv[]) {
    // Parameters
    double eps = 0.001;
    double shift_damping = 0.05;
    unsigned int shift_interval = 10;
    double en_shift = 0;
    unsigned int n_orb = 22;
    unsigned int n_frz = 2;
    unsigned int n_elec = 10;
    
    
    unsigned int n_elec_unf = n_elec - n_frz;
    unsigned int tot_orb = n_orb + n_frz / 2;
    
    // Read in data files
    unsigned char *symm = malloc(sizeof(unsigned char) * tot_orb);
    read_in_uchar(symm, "symm.txt");
    symm = &symm[n_frz / 2];
    
    double (* h_core)[tot_orb] = malloc(sizeof(double) * tot_orb * tot_orb);
    read_in_doub((double *)h_core, "hcore.txt");
    
    double (* eris)[tot_orb][tot_orb][tot_orb] = malloc(sizeof(double) * tot_orb * tot_orb * tot_orb * tot_orb);
    read_in_doub((double *)eris, "eris.txt");
    
    double hf_en;
    FILE *hf_en_file = fopen("hf_en.txt", "r");
    fscanf(hf_en_file, "%lf", &hf_en);
    fclose(hf_en_file);
    
    // Setup output files
//    FILE *num_file = fopen("projnum.txt", "w");
//    FILE *den_file = fopen("projden.txt", "w");
//    FILE *shift_file = fopen("S.txt", "w");
//    FILE *walk_file = fopen("N.txt", "w");
    
    // Setup arrays for determinants, lookup tables, and rn generators
    long long *sol_dets = malloc(sizeof(long long) * max_n_dets);
    int *sol_vals = malloc(sizeof(int) * max_n_dets);
    double *sol_mel = malloc(sizeof(double) * max_n_dets);
    unsigned char occ_orbs[max_n_dets][n_elec_unf];
    unsigned char byte_nums[256];
    unsigned char byte_idx[256][8];
    gen_byte_table(byte_idx, byte_nums);
    unsigned char symm_lookup[n_irreps][n_orb + 1];
    gen_symm_lookup(symm, n_orb, n_irreps, symm_lookup);
    unsigned int unocc_symm_cts[n_irreps][2];
    int n_mt = 1;
    mt_struct **rngen_ptrs = get_mt_parameters_st(32, 521, 4172, 4172 - 1 + n_mt, (unsigned int) time(NULL), &n_mt);
    int i;
    for (i = 0; i < n_mt; i++) {
        sgenrand_mt((uint32_t) time(NULL), rngen_ptrs[i]);
    }
    
    long long hf_det = gen_hf_bitstring(n_orb, n_elec - n_frz);
    sol_dets[0] = hf_det;
    sol_vals[0] = 100;
    size_t n_dets = 1;
    
    gen_orb_list(sol_dets[0], n_elec_unf, byte_nums, byte_idx, occ_orbs[0]);
    sol_mel[0] = diag_matrel(occ_orbs[0], tot_orb, eris, h_core, n_frz, n_elec) - hf_en;
    
    // Calculate HF column vector for enegry estimator, and count # single/double excitations
    size_t max_num_doub = count_doub_nosymm(n_elec_unf, n_orb);
    long long *hf_dets = malloc(max_num_doub * sizeof(long long));
    double *hf_mel = malloc(max_num_doub * sizeof(double));
    size_t n_hf_doub = gen_hf_ex(hf_det, &occ_orbs[0][0], n_elec_unf, tot_orb, symm, eris, n_frz, hf_dets, hf_mel);
    size_t n_hf_sing = count_singex(hf_det, &occ_orbs[0][0], symm, n_orb, symm_lookup, n_elec_unf);
    double p_doub = (double) n_hf_doub / (n_hf_sing + n_hf_doub);
    
    // Setup hash table for determinants
    hash_table *det_hash = setup_ht(target_walkers, rngen_ptrs[0], 2 * n_orb);
    stack_s *det_stack = setup_stack(1000);
    unsigned long long hash_val;
    unsigned long long hf_hashes[n_hf_doub];
    size_t det_idx;
    for (det_idx = 0; det_idx < n_hf_doub; det_idx++) {
        gen_orb_list(hf_dets[det_idx], n_elec_unf, byte_nums, byte_idx, occ_orbs[n_dets]);
        hf_hashes[det_idx] = hash_fxn(occ_orbs[n_dets], n_elec_unf, det_hash->scrambler);
    }
    
    // Setup arrays to hold spawned walkers
    long long *spawn_dets = malloc(sizeof(long long) * target_walkers);
    int *spawn_vals = malloc(sizeof(int) * target_walkers);
    unsigned int max_spawn = 100000;
    unsigned char *spawn_orbs = malloc(sizeof(unsigned char) * 4 * max_spawn);
    double *spawn_probs = malloc(sizeof(double) * max_spawn);
    unsigned char (*sing_orbs)[2];
    unsigned char (*doub_orbs)[4];
    
    size_t n_spawn, walker_idx;
    ssize_t *idx_ptr;
    unsigned int n_walk, n_doub, n_sing;
    unsigned int one_norm, last_one_norm = 0;
    int spawn_walker, walk_sign, new_val;
    long long new_det;
    double matr_el;
    
    unsigned int iterat;
    for (iterat = 0; iterat < max_iter; iterat++) {
        n_spawn = 0;
        one_norm = 0;
        for (det_idx = 0; det_idx < n_dets; det_idx++) {
            n_walk = abs(sol_vals[det_idx]);
            if (n_walk == 0) {
                continue;
            }
            walk_sign = 1 - 2 * signbit(sol_vals[det_idx]);
            
            // spawning step
            count_symm_virt(unocc_symm_cts, occ_orbs[det_idx], n_elec_unf,
                            n_orb, n_irreps, symm_lookup, symm);
            n_doub = bin_sample(n_walk, p_doub, rngen_ptrs[0]);
            n_sing = n_walk - n_doub;
            
            if (n_doub > max_spawn || n_sing / 2 > max_spawn) {
                printf("Allocating more memory for spawning\n");
                max_spawn *= 2;
                free(spawn_orbs);
                free(spawn_probs);
                spawn_orbs = malloc(sizeof(unsigned char) * 4 * max_spawn);
                spawn_probs = malloc(sizeof(double) * max_spawn);
            }
            
            doub_orbs = (unsigned char (*)[4]) spawn_orbs;
            n_doub = doub_multin(sol_dets[det_idx], occ_orbs[det_idx], n_elec_unf, symm, n_orb, symm_lookup, unocc_symm_cts, n_doub, rngen_ptrs[0], doub_orbs, spawn_probs);
            
            for (walker_idx = 0; walker_idx < n_doub; walker_idx++) {
                matr_el = doub_matr_el_nosgn(doub_orbs[walker_idx], tot_orb, eris, n_frz);
                matr_el *= eps / spawn_probs[walker_idx] / p_doub;
                spawn_walker = round_binomially(matr_el, 1, rngen_ptrs[0]);
                
                if (spawn_walker != 0) {
                    new_det = sol_dets[det_idx];
                    spawn_walker *= -doub_det_parity(&new_det, doub_orbs[walker_idx]) * (sol_vals[det_idx] / (int)n_walk);
                    if (new_det == hf_det) {
                        sol_vals[0] += spawn_walker;
                    }
                    else {
                        spawn_dets[n_spawn] = new_det;
                        spawn_vals[n_spawn] = spawn_walker;
                        n_spawn++;
                    }
                }
            }
            
            sing_orbs = (unsigned char (*)[2]) spawn_orbs;
            n_sing = sing_multin(sol_dets[det_idx], occ_orbs[det_idx], n_elec_unf, symm, n_orb, symm_lookup, unocc_symm_cts, n_sing, rngen_ptrs[0], sing_orbs, spawn_probs);
            
            for (walker_idx = 0; walker_idx < n_sing; walker_idx++) {
                matr_el = sing_matr_el_nosgn(sing_orbs[walker_idx], occ_orbs[det_idx], tot_orb, eris, h_core, n_frz, n_elec_unf);
                matr_el *= eps / spawn_probs[walker_idx] / (1 - p_doub);
                spawn_walker = round_binomially(matr_el, 1, rngen_ptrs[0]);
                
                if (spawn_walker != 0) {
                    new_det = sol_dets[det_idx];
                    spawn_walker *= -sing_det_parity(&new_det, doub_orbs[walker_idx]) * (sol_vals[det_idx] / n_walk);
                    if (new_det == hf_det) {
                        sol_vals[0] += spawn_walker;
                    }
                    else {
                        spawn_dets[n_spawn] = new_det;
                        spawn_vals[n_spawn] = spawn_walker;
                        n_spawn++;
                    }
                }
            }
            
            // Death/cloning step
            matr_el = (1 - eps * (sol_mel[det_idx] - en_shift)) * walk_sign;
            new_val = round_binomially(matr_el, n_walk, rngen_ptrs[0]);
            if (new_val == 0 && sol_dets[det_idx] != hf_det) {
                hash_val = hash_fxn(occ_orbs[det_idx], n_elec_unf, det_hash->scrambler);
                del_ht(det_hash, sol_dets[det_idx], hash_val);
                push(det_stack, det_idx);
            }
            sol_vals[det_idx] = new_val;
            one_norm += abs(new_val);
        }
        // Annihilation step
        for (walker_idx = 0; walker_idx < n_spawn; walker_idx++) {
            new_det = spawn_dets[walker_idx];
            gen_orb_list(new_det, n_elec_unf, byte_nums, byte_idx, spawn_orbs);
            hash_val = hash_fxn(spawn_orbs, n_elec_unf, det_hash->scrambler);
            idx_ptr = read_ht(det_hash, new_det, hash_val, 1);
            if (*idx_ptr == -1) {
                *idx_ptr = pop(det_stack);
                if (*idx_ptr == -1) {
                    if (n_dets >= max_n_dets) {
                        fprintf(stderr, "Exceeded maximum length of determinant arrays\n");
                        return -1;
                    }
                    *idx_ptr = n_dets;
                    n_dets++;
                }
                sol_dets[*idx_ptr] = new_det;
                sol_vals[*idx_ptr] = 0;
                // copy occupied orbitals over
                for (n_doub = 0; n_doub < n_elec_unf; n_doub++) {
                    occ_orbs[*idx_ptr][n_doub] = spawn_orbs[n_doub];
                }
                sol_mel[*idx_ptr] = diag_matrel(spawn_orbs, tot_orb, eris, h_core, n_frz, n_elec) - hf_en;
            }
            one_norm -= abs(sol_vals[*idx_ptr]);
            sol_vals[*idx_ptr] += spawn_vals[walker_idx];
            if (sol_vals[*idx_ptr] == 0) {
                push(det_stack, *idx_ptr);
                del_ht(det_hash, new_det, hash_val);
            }
            else {
                one_norm += abs(sol_vals[*idx_ptr]);
            }
        }
        if ((iterat + 1) % shift_interval == 0) {
            adjust_shift(&en_shift, one_norm, &last_one_norm, target_walkers, shift_damping / eps / shift_interval);
        }
        matr_el = calc_est_num(sol_dets, sol_vals, hf_dets, hf_mel, n_hf_doub, det_hash, hf_hashes);
//        fprintf(num_file, "%lf\n", matr_el);
//        fprintf(den_file, "%d\n", sol_vals[0]);
//        fprintf(walk_file, "%u\n", one_norm);
//        fprintf(shift_file, "%lf\n", en_shift);
        printf("%6u, n walk: %3u, en est: %lf, shift: %lf\n", iterat, one_norm, matr_el / sol_vals[0], en_shift);
    }
    
    return 0;
}
