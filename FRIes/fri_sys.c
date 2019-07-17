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
    "fri_sys [options] [[--] args]",
    "fri_sys [options]",
    NULL,
};

//double calc_est_num(long long *vec_dets, double *vec_vals, long long *hf_dets,
//                    double *hf_mel, size_t num_hf, hash_table *vec_hash,
//                    unsigned long long *hf_hashes) {
//    size_t hf_idx;
//    ssize_t *ht_ptr;
//    double numer = 0;
//    for (hf_idx = 0; hf_idx < num_hf; hf_idx++) {
//        ht_ptr = read_ht(vec_hash, hf_dets[hf_idx], hf_hashes[hf_idx], 0);
//        if (ht_ptr) {
//            numer += hf_mel[hf_idx] * vec_vals[*ht_ptr];
//        }
//    }
//    return numer;
//}


int main(int argc, const char * argv[]) {
    const char *hf_path = NULL;
    const char *result_dir = "./";
    const char *load_dir = NULL;
    const char *ini_dir = NULL;
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
        OPT_INTEGER('p', "max_dets", &max_n_dets, "Maximum number of determinants on a single MPI process."),
        OPT_FLOAT('i', "initiator", &init_thresh, "Magnitude of vector element required to make the corresponding determinant an initiator."),
        OPT_STRING('l', "load_dir", &load_dir, "Directory from which to load checkpoint files from a previous calculation."),
        OPT_STRING('n', "ini_dir", &ini_dir, "Directory from which to read the initial vector for a new calculation."),
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
    parse_hf_input(hf_path, &in_data);
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
    unsigned int (*unocc_symm_cts)[n_irreps][2] = malloc(sizeof(int) * 2 * n_irreps * max_n_dets);
    double *sol_vals = malloc(sizeof(int) * max_n_dets);
    double *sol_mel = malloc(sizeof(double) * max_n_dets);
    int *keep_exact = malloc(sizeof(int) * max_n_dets);
    size_t *srt_arr = malloc(sizeof(size_t) * max_n_dets);
    unsigned char (*occ_orbs)[n_elec_unf] = malloc(sizeof(unsigned char) * max_n_dets * n_elec_unf);
    size_t loc_n_dets = 0; // number of vector elements on this processor
    int loc_n_nonz = 0;
    ssize_t *idx_ptr;
    byte_table *b_tab = gen_byte_table();
    unsigned char symm_lookup[n_irreps][n_orb + 1];
    gen_symm_lookup(symm, n_orb, n_irreps, symm_lookup);
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
    gen_orb_list(hf_det, b_tab, occ_orbs[0]);
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
        gen_orb_list(hf_dets[det_idx], b_tab, occ_orbs[loc_n_dets]);
        hf_hashes[det_idx] = hash_fxn(occ_orbs[loc_n_dets], n_elec_unf, det_hash->scrambler);
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
    
    // Initialize solution vector
    if (load_dir) {
        loc_n_dets = load_vec(load_dir, sol_dets, sol_vals, sizeof(double));
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
    else if (ini_dir) {
        if (proc_rank == 0) {
            long long *load_dets;
            double *load_vals;
            size_t load_size = 0;
#ifdef USE_MPI
            load_dets = (long long *)send_dets;
            load_vals = (double *)send_vals;
#else
            load_dets = sol_dets;
            load_vals = sol_vals;
#endif
            
            load_size = load_vec_txt(ini_dir, load_dets, load_vals, DOUB);
            
            for (proc_idx = 0; proc_idx < n_procs; proc_idx++) {
                n_spawn[proc_idx] = 0;
            }
            for (det_idx = 0; det_idx < load_size; det_idx++) {
                gen_orb_list(load_dets[det_idx], b_tab, occ_orbs[0]);
                hash_val = hash_fxn(occ_orbs[0], n_elec_unf, proc_scrambler);
                proc_idx = hash_val % n_procs;
                send_dets[proc_idx][n_spawn[proc_idx]] = load_dets[det_idx];
                send_vals[proc_idx][n_spawn[proc_idx]] = load_vals[det_idx];
                n_spawn[proc_idx]++;
            }
        }
#ifdef USE_MPI
        MPI_Scatterv(send_dets, n_spawn, displacements, MPI_LONG_LONG, sol_dets, max_n_dets, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        MPI_Scatterv(send_vals, n_spawn, displacements, MPI_DOUBLE, sol_vals, max_n_dets, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    }
    else {
        if (hf_proc == proc_rank) {
            sol_dets[0] = hf_det;
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
            gen_orb_list(sol_dets[loc_n_nonz], b_tab, occ_orbs[loc_n_nonz]);
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
        
        strcpy(file_path, result_dir);
        strcat(file_path, "params.txt");
        FILE *param_f = fopen(file_path, "w");
        fprintf(param_f, "FRI calculation\nHF path: %s\nepsilon (imaginary time step): %lf\nTarget norm %lf\nInitiator threshold: %lf\nMatrix nonzero: %u\nVector nonzero: %u\n", hf_path, eps, target_norm, init_thresh, matr_samp, target_nonz);
        if (load_dir) {
            fprintf(param_f, "Restarting calculation from %s\n", load_dir);
        }
        else if (ini_dir) {
            fprintf(param_f, "Initializing calculation from vector at path %s\n", ini_dir);
        }
        else {
            fprintf(param_f, "Initializing calculation from HF unit vector\n");
        }
        fclose(param_f);
    }
    
    double *subwt_mem = malloc(sizeof(double) * n_irreps * spawn_length);
    unsigned int *ndiv_vec = malloc(sizeof(unsigned int) * spawn_length);
    double *comp_vec1 = malloc(sizeof(double) * spawn_length);
    double *comp_vec2 = malloc(sizeof(double) * spawn_length);
    size_t (*comp_idx)[2] = malloc(sizeof(size_t) * 2 * spawn_length);
    size_t comp_len;
    double *prob_vec1 = malloc(sizeof(double) * spawn_length);
    double *prob_vec2 = malloc(sizeof(double) * spawn_length);
    size_t *det_indices1 = malloc(sizeof(size_t) * spawn_length);
    size_t *det_indices2 = malloc(sizeof(size_t) * spawn_length);
    unsigned char (*orb_indices1)[5] = malloc(sizeof(char) * 5 * spawn_length);
    unsigned char (*orb_indices2)[5] = malloc(sizeof(char) * 5 * spawn_length);
    int *keep_idx = calloc(n_irreps * spawn_length, sizeof(int));
    double *wt_remain = calloc(spawn_length, sizeof(double));
    double loc_mat_norms[n_procs];
    unsigned int n_samp;
    size_t samp_idx;
    
    size_t walker_idx;
    long long ini_flag;
    unsigned int n_doub;
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
        sum_mpi_i(loc_n_nonz, &glob_n_nonz, proc_rank, n_procs);
        
        // Systematic matrix compression
        if (glob_n_nonz > matr_samp) {
            fprintf(stderr, "Warning: target number of matrix samples (%u) is less than number of nonzero vector elements (%d)\n", matr_samp, glob_n_nonz);
        }
        
        // Singles vs doubles
        for (det_idx = 0; det_idx < loc_n_dets; det_idx++) {
            weight = fabs(sol_vals[det_idx]);
            comp_vec1[det_idx] = weight;
            if (weight > 0) {
                subwt_mem[det_idx * 2] = p_doub;
                subwt_mem[det_idx * 2 + 1] = (1 - p_doub);
                ndiv_vec[det_idx] = 0;
                count_symm_virt(unocc_symm_cts[det_idx], occ_orbs[det_idx], n_elec_unf,
                                n_orb, n_irreps, symm_lookup, symm);
            }
        }
        n_samp = matr_samp;
        loc_mat_norms[proc_rank] = find_keep_sub(comp_vec1, ndiv_vec, 2, (double (*)[2])subwt_mem, (int (*)[2])keep_idx, loc_n_dets, &n_samp, wt_remain);
        
        if (proc_rank == 0) {
            rn_sys = genrand_mt(rngen_ptr) / MT_MAX;
        }
#ifdef USE_MPI
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, loc_mat_norms, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Bcast(&rn_sys, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
        comp_len = sys_sub(comp_vec1, ndiv_vec, 2, (double (*)[2])subwt_mem, (int (*)[2])keep_idx, loc_n_dets, n_samp, wt_remain, loc_mat_norms, rn_sys, comp_vec2, comp_idx);
        
        // Occupied orbital (single); occupied pair (double)
        for (samp_idx = 0; samp_idx < comp_len; samp_idx++) {
            det_idx = comp_idx[samp_idx][0];
            det_indices1[samp_idx] = det_idx;
            orb_indices1[samp_idx][0] = comp_idx[samp_idx][1];
            if (orb_indices1[samp_idx][0] == 0) { // double excitation
                ndiv_vec[samp_idx] = n_elec_unf * (n_elec_unf - 1) / 2;
            }
            else {
                ndiv_vec[samp_idx] = count_sing_allowed(occ_orbs[det_idx], n_elec_unf, symm, n_orb, unocc_symm_cts[det_idx]);
            }
        }
        
        n_samp = matr_samp;
        loc_mat_norms[proc_rank] = find_keep_sub(comp_vec2, ndiv_vec, 1, (double (*)[1])subwt_mem, (int (*)[1])keep_idx, comp_len, &n_samp, wt_remain);
        if (proc_rank == 0) {
            rn_sys = genrand_mt(rngen_ptr) / MT_MAX;
        }
#ifdef USE_MPI
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, loc_mat_norms, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Bcast(&rn_sys, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
        comp_len = sys_sub(comp_vec2, ndiv_vec, 1, (double (*)[1])subwt_mem, (int (*)[1])keep_idx, comp_len, n_samp, wt_remain, loc_mat_norms, rn_sys, comp_vec1, comp_idx);
        
        // Unoccupied orbital (single); 1st unoccupied (double)
        for (samp_idx = 0; samp_idx < comp_len; samp_idx++) {
            det_idx = comp_idx[samp_idx][0];
            det_indices2[samp_idx] = det_indices1[det_idx];
            orb_indices2[samp_idx][0] = orb_indices1[det_idx][0]; // single or double
            orb_indices2[samp_idx][1] = comp_idx[samp_idx][1]; // occupied orbital index or occupied pair index
            if (orb_indices2[samp_idx][0] == 0) { // double excitation
                <#statements#>
            }
            else { // single excitation
                ndiv_vec[samp_idx] = count_sing_virt(occ_orbs[det_idx], n_elec_unf, symm, n_orb, unocc_symm_cts[det_idx], &orb_indices2[samp_idx][1]);
            }
        }
        
        // Death/cloning step
        for (det_idx = 0; det_idx < loc_n_dets; det_idx++) {
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
                ini_flag = new_det & ini_bit; // came from initiator
                new_det &= ini_mask;
                gen_orb_list(new_det, b_tab, tmp_orbs);
                hash_val = hash_fxn(tmp_orbs, n_elec_unf, det_hash->scrambler);
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
                        occ_orbs[*idx_ptr][n_doub] = tmp_orbs[n_doub];
                    }
                    sol_mel[*idx_ptr] = diag_matrel(tmp_orbs, tot_orb, eris, h_core, n_frz, n_elec) - hf_en;
                }
                if (ini_flag || (idx_ptr && (signbit(sol_vals[*idx_ptr]) == signbit(recv_vals[proc_idx][walker_idx])))) {
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
                //                fprintf(shift_file, "%lf\n", en_shift);
                //                fprintf(norm_file, "%lf\n", glob_norm);
            }
        }
        matr_el = calc_dprod(sol_dets, sol_vals, hf_dets, hf_mel, n_hf_doub, det_hash, hf_hashes, DOUB);
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
            //            fprintf(num_file, "%lf\n", matr_el);
            //            fprintf(den_file, "%lf\n", sol_vals[0]);
            printf("%6u, en est: %lf, shift: %lf, norm: %lf\n", iterat, matr_el / sol_vals[0], en_shift, glob_norm);
        }
        
        if (proc_rank == 0) {
            rn_sys = genrand_mt(rngen_ptr) / MT_MAX;
        }
#ifdef USE_MPI
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, loc_norms, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Bcast(&rn_sys, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
        sys_comp(sol_vals, loc_n_dets, loc_norms, n_samp, keep_exact, rn_sys);
        for (det_idx = 0; det_idx < loc_n_dets; det_idx++) {
            if (keep_exact[det_idx] && sol_dets[det_idx] != hf_det) {
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



