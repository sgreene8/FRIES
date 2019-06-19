#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "io_utils.h"
#include "fci_utils.h"
#include "near_uniform.h"
#include "dc.h"
#include "compress_utils.h"
#include "det_store.h"
#include "heap.h"
#include "argparse.h"
#include "mpi_switch.h"
#define max_iter 10

static const char *const usage[] = {
    "fri [options] [[--] args]",
    "fri [options]",
    NULL,
};

double calc_est_num(long long *vec_dets, double *vec_vals, long long *hf_dets,
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
    const char *hf_path = NULL;
    const char *result_dir = "./";
    const char *load_dir = NULL;
    double target_norm = 0;
    unsigned int target_nonz = 0;
    unsigned int matr_samp = 0;
    unsigned int max_n_dets = 0;
    double init_thresh = 0;
    struct argparse_option options[] = {
        OPT_HELP(),
        OPT_STRING('d', "hf_path", &hf_path, "Path to the directory that contains the HF output files eris.txt, hcore.txt, symm.txt, hf_en.txt, and sys_params.txt"),
        OPT_FLOAT('t', "target", &target_norm, "Target one-norm of solution vector"),
        OPT_INTEGER('m', "vec_nonz", &target_nonz, "Target number of nonzero vector elements to keep after each iteration"),
        OPT_INTEGER('M', "mat_nonz", &matr_samp, "Target number of nonzero matrix elements to keep after each iteration"),
        OPT_STRING('y', "result_dir", &result_dir, "Directory in which to save output files"),
        OPT_INTEGER('n', "max_dets", &max_n_dets, "Maximum number of determinants on a single MPI process."),
        OPT_FLOAT('i', "initiator", &init_thresh, "Magnitude of vector element required to make the corresponding determinant an initiator."),
        OPT_STRING('l', "load_dir", &load_dir, "Directory from which to load the vector to initialize the calculation."),
        OPT_END(),
    };
    
    struct argparse argparse;
    argparse_init(&argparse, options, usage, 0);
    argparse_describe(&argparse, "\nPerform an FCIQMC calculation.", "");
    argc = argparse_parse(&argparse, argc, argv);
    
    if (hf_path == NULL) {
        fprintf(stderr, "Error: HF directory not specified.\n");
        return 0;
    }
    if (target_nonz == 0) {
        fprintf(stderr, "Error: target number of nonzero vector elements not specified\n");
        return 0;
    }
    if (matr_samp == 0) {
        fprintf(stderr, "Error: target number of nonzero matrix elements not specified\n");
        return 0;
    }
    if (max_n_dets == 0) {
        fprintf(stderr, "Error: maximum number of determinants expected on each processor not specified.\n");
        return 0;
    }
    init_thresh = fabs(init_thresh);
    
    int n_procs = 1;
    int proc_rank = 0;
    unsigned int proc_idx, hf_proc;
#ifdef USE_MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
#endif
    
    // Parameters
    double shift_damping = 0.05;
    unsigned int shift_interval = 10;
    unsigned int save_interval = 1000;
    double en_shift = 0;
    
    // Read in data files
    hf_input in_data;
    parse_input(hf_path, &in_data);
    double eps = in_data.eps;
    unsigned int n_elec = in_data.n_elec;
    unsigned int n_frz = in_data.n_frz;
    unsigned int n_orb = in_data.n_orb;
    double hf_en = in_data.hf_en;;
    
    unsigned int n_elec_unf = n_elec - n_frz;
    unsigned int tot_orb = n_orb + n_frz / 2;
    long long ini_bit = 1LL << (2 * n_orb);
    long long ini_mask = ini_bit - 1;
    
    unsigned char *symm = in_data.symm;
    double (* h_core)[tot_orb] = (double (*)[tot_orb])in_data.hcore;
    double (* eris)[tot_orb][tot_orb][tot_orb] = (double (*)[tot_orb][tot_orb][tot_orb])in_data.eris;
    
    // Setup arrays for determinants, lookup tables, and rn generators
    long long *sol_dets = malloc(sizeof(long long) * max_n_dets);
    double *sol_vals = malloc(sizeof(int) * max_n_dets);
    double *sol_mel = malloc(sizeof(double) * max_n_dets);
    int *keep_exact = malloc(sizeof(int) * max_n_dets);
    size_t *srt_arr = malloc(sizeof(size_t) * max_n_dets);
    unsigned char (*occ_orbs)[n_elec_unf] = malloc(sizeof(unsigned char) * max_n_dets * n_elec_unf);
    size_t loc_n_dets = 0; // number of vector elements on this processor
    int loc_n_nonz = 0;
    ssize_t *idx_ptr;
    unsigned char byte_nums[256];
    unsigned char byte_idx[256][8];
    gen_byte_table(byte_idx, byte_nums);
    unsigned char symm_lookup[n_irreps][n_orb + 1];
    gen_symm_lookup(symm, n_orb, n_irreps, symm_lookup);
    unsigned int unocc_symm_cts[n_irreps][2];
    mt_struct *rngen_ptr = get_mt_parameter_id_st(32, 607, proc_rank, (unsigned int) time(NULL));
    sgenrand_mt((uint32_t) time(NULL), rngen_ptr);
    
    // Setup hash table for determinants
    hash_table *det_hash = setup_ht(max_n_dets / n_procs, rngen_ptr, 2 * n_orb);
    stack_s *det_stack = setup_stack(1000);
    unsigned long long hash_val;
    
    // Initialize hash function for processors and vector
    size_t det_idx;
    unsigned int proc_scrambler[2 * n_orb];
    long long hf_det = gen_hf_bitstring(n_orb, n_elec - n_frz);
    double loc_norms[n_procs];
    double glob_norm;
    
    if (load_dir) {
        load_proc_hash(load_dir, proc_scrambler);
    }
    else {
        if (proc_rank == 0) {
            for (det_idx = 0; det_idx < 2 * n_orb; det_idx++) {
                proc_scrambler[det_idx] = genrand_mt(rngen_ptr);
            }
            save_proc_hash(result_dir, proc_scrambler, 2 * n_orb);
        }
#ifdef USE_MPI
        MPI_Bcast(&proc_scrambler, 2 * n_orb, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
#endif
    }
    gen_orb_list(hf_det, n_elec_unf, byte_nums, byte_idx, occ_orbs[0]);
    hash_val = hash_fxn(occ_orbs[0], n_elec_unf, proc_scrambler);
    hf_proc = hash_val % n_procs;
    
    // Calculate HF column vector for enegry estimator, and count # single/double excitations
    size_t max_num_doub = count_doub_nosymm(n_elec_unf, n_orb);
    long long *hf_dets = malloc(max_num_doub * sizeof(long long));
    double *hf_mel = malloc(max_num_doub * sizeof(double));
    size_t n_hf_doub = gen_hf_ex(hf_det, occ_orbs[0], n_elec_unf, tot_orb, symm, eris, n_frz, hf_dets, hf_mel);
    size_t n_hf_sing = count_singex(hf_det, occ_orbs[0], symm, n_orb, symm_lookup, n_elec_unf);
    double p_doub = (double) n_hf_doub / (n_hf_sing + n_hf_doub);
    
    unsigned long long hf_hashes[n_hf_doub];
    for (det_idx = 0; det_idx < n_hf_doub; det_idx++) {
        gen_orb_list(hf_dets[det_idx], n_elec_unf, byte_nums, byte_idx, occ_orbs[loc_n_dets]);
        hf_hashes[det_idx] = hash_fxn(occ_orbs[loc_n_dets], n_elec_unf, det_hash->scrambler);
    }
    
    // Initialize solution vector
    if (load_dir) {
        loc_n_dets = load_vec(load_dir, sol_dets, sol_vals, sizeof(double));
    }
    else {
        if (hf_proc == proc_rank) {
            sol_dets[0] = hf_det | ini_bit;
            sol_vals[0] = 100;
            loc_n_dets = 1;
        }
        else {
            loc_n_dets = 0;
        }
    }
    loc_norms[proc_rank] = 0;
    for (det_idx = 0; det_idx < loc_n_dets; det_idx++) {
        if (sol_vals[det_idx] != 0) {
            sol_dets[loc_n_nonz] = sol_dets[det_idx];
            sol_vals[loc_n_nonz] = sol_vals[det_idx];
            gen_orb_list(sol_dets[loc_n_nonz], n_elec_unf, byte_nums, byte_idx, occ_orbs[loc_n_nonz]);
            sol_mel[loc_n_nonz] = diag_matrel(occ_orbs[loc_n_nonz], tot_orb, eris, h_core, n_frz, n_elec) - hf_en;
            hash_val = hash_fxn(occ_orbs[loc_n_nonz], n_elec_unf, det_hash->scrambler);
            idx_ptr = read_ht(det_hash, sol_dets[loc_n_nonz], hash_val, 1);
            *idx_ptr = loc_n_nonz;
            loc_norms[proc_rank] += fabs(sol_vals[det_idx]);
            keep_exact[loc_n_nonz] = 0;
            srt_arr[loc_n_nonz] = loc_n_nonz;
            loc_n_nonz++;
        }
    }
    loc_n_dets = loc_n_nonz;
#ifdef USE_MPI
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, loc_norms, 1, MPI_DOUBLE, MPI_COMM_WORLD);
#endif
    glob_norm = 0;
    for (proc_idx = 0; proc_idx < n_procs; proc_idx++) {
        glob_norm += loc_norms[proc_idx];
    }
    
    char file_path[100];
    FILE *num_file = NULL;
    FILE *den_file = NULL;
    FILE *shift_file = NULL;
    FILE *norm_file = NULL;
    if (proc_rank == hf_proc) {
        // Setup output files
        strcpy(file_path, result_dir);
        strcat(file_path, "projnum.txt");
        num_file = fopen(file_path, "a");
        strcpy(file_path, result_dir);
        strcat(file_path, "projden.txt");
        den_file = fopen(file_path, "a");
        strcpy(file_path, result_dir);
        strcat(file_path, "S.txt");
        shift_file = fopen(file_path, "a");
        strcpy(file_path, result_dir);
        strcat(file_path, "norm.txt");
        norm_file = fopen(file_path, "a");
    }
    
    // Setup arrays to hold spawned walkers
    unsigned int spawn_length = matr_samp * 2 / n_procs;
    long long (*send_dets)[spawn_length] = malloc(sizeof(long long) * spawn_length * n_procs);
    double (*send_vals)[spawn_length] = malloc(sizeof(double) * spawn_length * n_procs);
    long long (*recv_dets)[spawn_length] = malloc(sizeof(long long) * spawn_length * n_procs);
    double (*recv_vals)[spawn_length] = malloc(sizeof(double) * spawn_length * n_procs);
    int n_spawn[n_procs];
    int recv_cts[n_procs];
    int displacements[n_procs];
    for (proc_idx = 0; proc_idx < n_procs; proc_idx++) {
        displacements[proc_idx] = proc_idx * spawn_length;
    }
    
    unsigned int max_spawn = matr_samp; // should scale as max expected # from one determinant
    unsigned char *spawn_orbs = malloc(sizeof(unsigned char) * 4 * max_spawn);
    double *spawn_probs = malloc(sizeof(double) * max_spawn);
    unsigned char (*sing_orbs)[2];
    unsigned char (*doub_orbs)[4];
    
    size_t walker_idx;
    long long ini_flag;
    unsigned int n_walk, n_doub, n_sing;
    double last_one_norm = 0;
    unsigned char tmp_orbs[n_elec_unf];
    long long new_det;
    double matr_el;
    double recv_nums[n_procs];
    
    // Parameters for systematic sampling
    double rn_sys = 0;
    double lbound;
    double weight;
    int glob_n_nonz; // Number of nonzero elements in whole vector (across all processors)
    
    unsigned int iterat;
    for (iterat = 0; iterat < max_iter; iterat++) {
        for (proc_idx = 0; proc_idx < n_procs; proc_idx++) {
            n_spawn[proc_idx] = 0;
        }
        //        printf("loc_n_nonz from proc %d/%u: %d\n", proc_rank, n_procs, loc_n_nonz);
        sum_mpi_i(loc_n_nonz, &glob_n_nonz, proc_rank, n_procs);
        //        printf("glob_n_nonz: %d\n", glob_n_nonz);
        
        // Systematic sampling to determine number of samples for each column
        if (proc_rank == 0) {
            rn_sys = genrand_mt(rngen_ptr) / MT_MAX;
        }
#ifdef USE_MPI
        MPI_Bcast(&rn_sys, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
        if (glob_n_nonz < matr_samp) {
            lbound = seed_sys(loc_norms, &rn_sys, matr_samp - (unsigned int)glob_n_nonz);
        }
        else {
            fprintf(stderr, "Warning: target number of matrix samples (%u) is less than number of nonzero vector elements (%d)\n", matr_samp, glob_n_nonz);
            lbound = 0;
            rn_sys = INFINITY;
        }
        for (det_idx = 0; det_idx < loc_n_dets; det_idx++) {
            weight = fabs(sol_vals[det_idx]);
            if (weight == 0) {
                continue;
            }
            n_walk = 1;
            lbound += weight;
            while (rn_sys < lbound) {
                n_walk++;
                rn_sys += glob_norm / (matr_samp - glob_n_nonz);
            }
            
            ini_flag = weight > init_thresh;
            ini_flag <<= 2 * n_orb;
            sol_dets[det_idx] = ini_flag | (sol_dets[det_idx] & ini_mask);
            
            // spawning step
            count_symm_virt(unocc_symm_cts, occ_orbs[det_idx], n_elec_unf,
                            n_orb, n_irreps, symm_lookup, symm);
            n_doub = bin_sample(n_walk, p_doub, rngen_ptr);
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
            n_doub = doub_multin(sol_dets[det_idx], occ_orbs[det_idx], n_elec_unf, symm, n_orb, symm_lookup, unocc_symm_cts, n_doub, rngen_ptr, doub_orbs, spawn_probs);
            
            for (walker_idx = 0; walker_idx < n_doub; walker_idx++) {
                matr_el = doub_matr_el_nosgn(doub_orbs[walker_idx], tot_orb, eris, n_frz);
                if (fabs(matr_el) > 1e-9) {
                    new_det = sol_dets[det_idx];
                    matr_el *= -eps / spawn_probs[walker_idx] / p_doub / n_walk * sol_vals[det_idx] * doub_det_parity(&new_det, doub_orbs[walker_idx]);
                    
                    gen_orb_list(new_det & ini_mask, n_elec_unf, byte_nums, byte_idx, tmp_orbs);
                    hash_val = hash_fxn(tmp_orbs, n_elec_unf, proc_scrambler);
                    proc_idx = hash_val % n_procs;
                    send_dets[proc_idx][n_spawn[proc_idx]] = new_det;
                    send_vals[proc_idx][n_spawn[proc_idx]] = matr_el;
                    n_spawn[proc_idx]++;
                }
            }
            
            sing_orbs = (unsigned char (*)[2]) spawn_orbs;
            n_sing = sing_multin(sol_dets[det_idx], occ_orbs[det_idx], n_elec_unf, symm, n_orb, symm_lookup, unocc_symm_cts, n_sing, rngen_ptr, sing_orbs, spawn_probs);
            
            for (walker_idx = 0; walker_idx < n_sing; walker_idx++) {
                matr_el = sing_matr_el_nosgn(sing_orbs[walker_idx], occ_orbs[det_idx], tot_orb, eris, h_core, n_frz, n_elec_unf);
                if (fabs(matr_el) > 1e-9) {
                    new_det = sol_dets[det_idx];
                    matr_el *= -eps / spawn_probs[walker_idx] / (1 - p_doub) / n_walk * sol_vals[det_idx] * sing_det_parity(&new_det, sing_orbs[walker_idx]);
                    
                    gen_orb_list(new_det & ini_mask, n_elec_unf, byte_nums, byte_idx, tmp_orbs);
                    hash_val = hash_fxn(tmp_orbs, n_elec_unf, proc_scrambler);
                    proc_idx = hash_val % n_procs;
                    send_dets[proc_idx][n_spawn[proc_idx]] = new_det;
                    send_vals[proc_idx][n_spawn[proc_idx]] = matr_el;
                    n_spawn[proc_idx]++;
                }
            }
            
            // Death/cloning step
            sol_vals[det_idx] *= 1 - eps * (sol_mel[det_idx] - en_shift);
        }
        // Communication
#ifdef USE_MPI
        MPI_Alltoall(n_spawn, 1, MPI_INT, recv_cts, 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Alltoallv(send_dets, n_spawn, displacements, MPI_LONG_LONG, recv_dets, recv_cts, displacements, MPI_LONG_LONG, MPI_COMM_WORLD);
        MPI_Alltoallv(send_vals, n_spawn, displacements, MPI_DOUBLE, recv_vals, recv_cts, displacements, MPI_DOUBLE, MPI_COMM_WORLD);
#else
        for (proc_idx = 0; proc_idx < n_procs; proc_idx++) {
            recv_cts[proc_idx] = n_spawn[proc_idx];
            for (walker_idx = 0; walker_idx < n_spawn[proc_idx]; walker_idx++) {
                recv_dets[proc_idx][walker_idx] = send_dets[proc_idx][walker_idx];
                recv_vals[proc_idx][walker_idx] = send_vals[proc_idx][walker_idx];
            }
        }
#endif
        // Annihilation step
        for (proc_idx = 0; proc_idx < n_procs; proc_idx++){
            for (walker_idx = 0; walker_idx < recv_cts[proc_idx]; walker_idx++) {
                new_det = recv_dets[proc_idx][walker_idx];
                gen_orb_list(new_det & ini_mask, n_elec_unf, byte_nums, byte_idx, spawn_orbs);
                hash_val = hash_fxn(spawn_orbs, n_elec_unf, det_hash->scrambler);
                ini_flag = new_det & ini_bit; // came from initiator
                new_det &= ini_mask;
                idx_ptr = read_ht(det_hash, new_det, hash_val, !(!ini_flag));
                if (idx_ptr && *idx_ptr == -1) {
                    *idx_ptr = pop(det_stack);
                    if (*idx_ptr == -1) {
                        if (loc_n_dets >= max_n_dets) {
                            fprintf(stderr, "Exceeded maximum length of determinant arrays\n");
                            return -1;
                        }
                        *idx_ptr = loc_n_dets;
                        keep_exact[loc_n_dets] = 0;
                        srt_arr[loc_n_dets] = loc_n_dets;
                        loc_n_dets++;
                    }
                    sol_dets[*idx_ptr] = new_det;
                    sol_vals[*idx_ptr] = 0;
                    loc_n_nonz++;
                    // copy occupied orbitals over
                    for (n_doub = 0; n_doub < n_elec_unf; n_doub++) {
                        occ_orbs[*idx_ptr][n_doub] = spawn_orbs[n_doub];
                    }
                    sol_mel[*idx_ptr] = diag_matrel(spawn_orbs, tot_orb, eris, h_core, n_frz, n_elec) - hf_en;
                }
                if (ini_flag || (idx_ptr && fabs(sol_vals[*idx_ptr]) > 0)) {
                    sol_vals[*idx_ptr] += recv_vals[proc_idx][walker_idx];
                }
            }
        }
        // Compression step
        unsigned int n_samp = target_nonz;
        loc_norms[proc_rank] = find_preserve(sol_vals, srt_arr, keep_exact, loc_n_dets, &n_samp, &glob_norm);
        
        // Adjust shift
        if ((iterat + 1) % shift_interval == 0) {
            adjust_shift(&en_shift, glob_norm, &last_one_norm, target_norm, shift_damping / shift_interval / eps);
            if (proc_rank == hf_proc) {
                fprintf(shift_file, "%lf\n", en_shift);
                fprintf(norm_file, "%lf\n", glob_norm);
            }
        }
        matr_el = calc_est_num(sol_dets, sol_vals, hf_dets, hf_mel, n_hf_doub, det_hash, hf_hashes);
#ifdef USE_MPI
        MPI_Gather(&matr_el, 1, MPI_DOUBLE, recv_nums, 1, MPI_DOUBLE, hf_proc, MPI_COMM_WORLD);
#else
        recv_nums[0] = matr_el;
#endif
        if (proc_rank == hf_proc) {
            matr_el = 0;
            for (proc_idx = 0; proc_idx < n_procs; proc_idx++) {
                matr_el += recv_nums[proc_idx];
            }
            fprintf(num_file, "%lf\n", matr_el);
            fprintf(den_file, "%lf\n", sol_vals[0]);
//            printf("%6u, en est: %lf, shift: %lf, norm: %lf\n", iterat, matr_el / sol_vals[0], en_shift, glob_norm);
        }
        
        if (proc_rank == 0) {
            rn_sys = genrand_mt(rngen_ptr) / MT_MAX;
        }
#ifdef USE_MPI
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, loc_norms, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Bcast(&rn_sys, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
        sys_solvec(sol_vals, loc_n_dets, loc_norms, n_samp, keep_exact, rn_sys);
        for (det_idx = 0; det_idx < loc_n_dets; det_idx++) {
            if (keep_exact[det_idx]) {
                push(det_stack, det_idx);
                hash_val = hash_fxn(occ_orbs[det_idx], n_elec_unf, det_hash->scrambler);
                del_ht(det_hash, sol_dets[det_idx], hash_val);
                keep_exact[det_idx] = 0;
                loc_n_nonz--;
            }
        }
        
        if ((iterat + 1) % save_interval == 0) {
            save_vec(result_dir, sol_dets, sol_vals, loc_n_dets, sizeof(double));
            if (proc_rank == hf_proc) {
                fflush(num_file);
                fflush(den_file);
                fflush(shift_file);
            }
        }
    }
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}
