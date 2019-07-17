/*
 Perform FCIQMC on the Hubbard model.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "io_utils.h"
#include "dc.h"
#include "compress_utils.h"
#include "det_store.h"
#include "argparse.h"
#include "mpi_switch.h"
#include "hub_holstein.h"
#define max_iter 10000

static const char *const usage[] = {
    "fciqmc_hh [options] [[--] args]",
    "fciqmc_hh [options]",
    NULL,
};

int pow_int(int base, int exp) {
    unsigned int idx;
    int result = 1;
    for (idx = 0; idx < exp; idx++) {
        result *= base;
    }
    return result;
}

int main(int argc, const char * argv[]) {
    int n_procs = 1;
    int proc_rank = 0;
    unsigned int proc_idx, ref_proc;
#ifdef USE_MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
#endif
    
    const char *params_path = NULL;
    const char *result_dir = "./";
    const char *load_dir = NULL;
    const char *ini_dir = NULL;
    unsigned int target_walkers = 0;
    unsigned int max_n_dets = 0;
    unsigned int init_thresh = 0;
    struct argparse_option options[] = {
        OPT_HELP(),
        OPT_STRING('d', "params_path", &params_path, "Path to the file that contains the parameters defining the Hamiltonian, number of electrons, number of sites, etc."),
        OPT_INTEGER('t', "target", &target_walkers, "Target number of walkers, must be greater than the plateau value for this system"),
        OPT_STRING('y', "result_dir", &result_dir, "Directory in which to save output files"),
        OPT_INTEGER('p', "max_dets", &max_n_dets, "Maximum number of determinants on a single MPI process."),
        OPT_INTEGER('i', "initiator", &init_thresh, "Number of walkers on a determinant required to make it an initiator."),
        OPT_STRING('l', "load_dir", &load_dir, "Directory from which to load checkpoint files from a previous calculation."),
        OPT_STRING('n', "ini_dir", &ini_dir, "Directory from which to read the initial vector for a new calculation."),
        OPT_END(),
    };
    
    struct argparse argparse;
    argparse_init(&argparse, options, usage, 0);
    argparse_describe(&argparse, "\nPerform an FCIQMC calculation.", "");
    argc = argparse_parse(&argparse, argc, argv);
    
    if (params_path == NULL) {
        fprintf(stderr, "Error: parameter file not specified.\n");
        return 0;
    }
    if (target_walkers == 0) {
        fprintf(stderr, "Error: target number of walkers not specified\n");
        return 0;
    }
    double target_norm = target_walkers;
    if (max_n_dets == 0) {
        fprintf(stderr, "Error: maximum number of determinants expected on each processor not specified.\n");
        return 0;
    }
    
    // Parameters
    double shift_damping = 0.05;
    unsigned int shift_interval = 10;
    unsigned int save_interval = 1000;
    double en_shift = 0;
    
    // Read in data files
    hh_input in_data;
    parse_hh_input(params_path, &in_data);
    double eps = in_data.eps;
    unsigned int hub_len = in_data.lat_len;
    unsigned int hub_dim = in_data.n_dim;
    unsigned int n_elec = in_data.n_elec;
    double hub_t = 1;
    double hub_u = in_data.elec_int;
    double hf_en = in_data.hf_en;
    
    if (hub_dim != 1) {
        fprintf(stderr, "Error: only 1-D Hubbard calculations supported right now.\n");
        return 0;
    }
    
    unsigned int n_orb = pow_int(hub_len, hub_dim);
    long long ini_bit = 1LL << (2 * n_orb);
    long long ini_mask = ini_bit - 1;
    
    // Setup arrays for determinants, lookup tables, and rn generators
    long long *sol_dets = malloc(sizeof(long long) * max_n_dets);
    int *sol_vals = malloc(sizeof(int) * max_n_dets);
    double *sol_mel = malloc(sizeof(double) * max_n_dets);
    unsigned char (*neighb_orbs)[2][n_elec + 1] = malloc(sizeof(unsigned char) * 2 * max_n_dets * (n_elec + 1));
    size_t n_dets = 0; // number of nonzero vector elements on this processor
    ssize_t *idx_ptr;
    byte_table *b_tab = gen_byte_table();
    mt_struct *rngen_ptr = get_mt_parameter_id_st(32, 521, proc_rank, (unsigned int) time(NULL));
    sgenrand_mt((uint32_t) time(NULL), rngen_ptr);
    
    // Setup hash table for determinants
    hash_table *det_hash = setup_ht(max_n_dets, rngen_ptr, 2 * n_orb);
    stack_s *det_stack = setup_stack(1000);
    unsigned long long hash_val;
    
    // Initialize hash function for processors and vector
    size_t det_idx;
    unsigned int proc_scrambler[2 * n_orb];
    long long neel_det = gen_hub_bitstring(n_orb, n_elec, hub_dim);
    int loc_norm, glob_norm;
    double last_norm = 0;
    
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
    find_neighbors(neel_det, hub_len, b_tab, n_elec, neighb_orbs[0]);
    hash_val = hash_fxn_hh(n_elec, neighb_orbs[0], proc_scrambler);
    ref_proc = hash_val % n_procs;
    
    // Setup arrays to hold spawned walkers
    unsigned int spawn_length = target_walkers / n_procs / n_procs;
    long long (*send_dets)[spawn_length] = malloc(sizeof(long long) * spawn_length * n_procs);
    int (*send_vals)[spawn_length] = malloc(sizeof(int) * spawn_length * n_procs);
    long long (*recv_dets)[spawn_length] = malloc(sizeof(long long) * spawn_length * n_procs);
    int (*recv_vals)[spawn_length] = malloc(sizeof(int) * spawn_length * n_procs);
    int n_spawn[n_procs];
    int recv_cts[n_procs];
    int displacements[n_procs];
    for (proc_idx = 0; proc_idx < n_procs; proc_idx++) {
        displacements[proc_idx] = proc_idx * spawn_length;
    }
    size_t walker_idx;
    
    // Initialize solution vector
    if (load_dir) {
        n_dets = load_vec(load_dir, sol_dets, sol_vals, sizeof(int));
    }
    else if (ini_dir) {
        long long *load_dets;
        int *load_vals;
#ifdef USE_MPI
        load_dets = (long long *) recv_dets;
        load_vals = (int *) recv_vals;
#else
        load_dets = sol_dets;
        load_vals = sol_vals;
#endif
        
        char buf[100];
        sprintf(buf, "%sqmc_", ini_dir);
        n_dets = load_vec_txt(buf, load_dets, load_vals, INT);
        
        for (proc_idx = 0; proc_idx < n_procs; proc_idx++) {
            n_spawn[proc_idx] = 0;
        }
        for (det_idx = 0; det_idx < n_dets; det_idx++) {
            find_neighbors(load_dets[det_idx], hub_len, b_tab, n_elec, neighb_orbs[0]);
            hash_val = hash_fxn_hh(n_elec, neighb_orbs[0], proc_scrambler);
            proc_idx = hash_val % n_procs;
            send_dets[proc_idx][n_spawn[proc_idx]] = load_dets[det_idx];
            send_vals[proc_idx][n_spawn[proc_idx]] = load_vals[det_idx];
            n_spawn[proc_idx]++;
        }
#ifdef USE_MPI
        MPI_Alltoall(n_spawn, 1, MPI_INT, recv_cts, 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Alltoallv(send_dets, n_spawn, displacements, MPI_LONG_LONG, recv_dets, recv_cts, displacements, MPI_LONG_LONG, MPI_COMM_WORLD);
        MPI_Alltoallv(send_vals, n_spawn, displacements, MPI_INT, recv_vals, recv_cts, displacements, MPI_INT, MPI_COMM_WORLD);
        
        n_dets = 0;
        for (proc_idx = 0; proc_idx < n_procs; proc_idx++){
            for (walker_idx = 0; walker_idx < recv_cts[proc_idx]; walker_idx++) {
                sol_dets[n_dets] = recv_dets[proc_idx][walker_idx];
                sol_vals[n_dets] = recv_vals[proc_idx][walker_idx];
                n_dets++;
            }
        }
#endif
    }
    else {
        if (ref_proc == proc_rank) {
            sol_dets[0] = neel_det;
            sol_vals[0] = 100;
            n_dets = 1;
        }
        else {
            n_dets = 0;
        }
    }
    loc_norm = 0;
    size_t n_nonz = 0;
    for (det_idx = 0; det_idx < n_dets; det_idx++) {
        if (sol_vals[det_idx] != 0) {
            sol_dets[n_nonz] = sol_dets[det_idx];
            sol_vals[n_nonz] = sol_vals[det_idx];
            sol_mel[n_nonz] = hub_u * hub_diag(sol_dets[det_idx], hub_len, b_tab);
            find_neighbors(sol_dets[det_idx], hub_len, b_tab, n_elec, neighb_orbs[n_nonz]);
            hash_val = hash_fxn_hh(n_elec, neighb_orbs[n_nonz], det_hash->scrambler);
            idx_ptr = read_ht(det_hash, sol_dets[det_idx], hash_val, 1);
            *idx_ptr = n_nonz;
            loc_norm += abs(sol_vals[det_idx]);
            n_nonz++;
        }
    }
    n_dets = n_nonz;
    sum_mpi_i(loc_norm, &glob_norm, proc_rank, n_procs);
    if (load_dir) {
        last_norm = glob_norm;
    }
    
    char file_path[100];
    FILE *walk_file = NULL;
    FILE *num_file = NULL;
    FILE *den_file = NULL;
    FILE *shift_file = NULL;
    FILE *nonz_file = NULL;
    if (proc_rank == ref_proc) {
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
        strcat(file_path, "N.txt");
        walk_file = fopen(file_path, "a");
        strcpy(file_path, result_dir);
        strcat(file_path, "nnonz.txt");
        nonz_file = fopen(file_path, "a");
        
        
        strcpy(file_path, result_dir);
        strcat(file_path, "params.txt");
        FILE *param_f = fopen(file_path, "w");
        fprintf(param_f, "FCIQMC calculation\nHubbard-Holstein parameters path: %s\nepsilon (imaginary time step): %lf\nTarget number of walkers %u\nInitiator threshold: %u\n", params_path, eps, target_walkers, init_thresh);
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
    
    unsigned int max_spawn = 500000; // should scale as max expected # on one determinant
    unsigned char (*spawn_orbs)[2] = malloc(sizeof(unsigned char) * 2 * max_spawn);
    
    long long ini_flag;
    unsigned int n_walk, n_success;
    int spawn_walker, walk_sign, new_val;
    unsigned char tmp_orbs[2][n_elec + 1];
    long long new_det;
    double matr_el;
    
    unsigned int iterat;
    int glob_nnonz;
    for (iterat = 0; iterat < max_iter; iterat++) {
        for (proc_idx = 0; proc_idx < n_procs; proc_idx++) {
            n_spawn[proc_idx] = 0;
        }
        n_nonz = 0;
        for (det_idx = 0; det_idx < n_dets; det_idx++) {
            n_walk = abs(sol_vals[det_idx]);
            if (n_walk == 0) {
                continue;
            }
            n_nonz++;
            ini_flag = n_walk > init_thresh;
            ini_flag <<= 2 * n_orb;
            walk_sign = 1 - ((sol_vals[det_idx] >> (sizeof(sol_vals[det_idx]) * 8 - 1)) & 2);
            
            // spawning step
            matr_el = eps * hub_t * (neighb_orbs[det_idx][0][0] + neighb_orbs[det_idx][1][0]);
            n_success = round_binomially(matr_el, n_walk, rngen_ptr);
            
            if (n_success > max_spawn) {
                printf("Allocating more memory for spawning\n");
                max_spawn *= 2;
                free(spawn_orbs);
                spawn_orbs = malloc(sizeof(unsigned char) * 2 * max_spawn);
            }
            
            hub_multin(sol_dets[det_idx], n_elec, neighb_orbs[det_idx], n_success, rngen_ptr, spawn_orbs);
            
            for (walker_idx = 0; walker_idx < n_success; walker_idx++) {
                new_det = sol_dets[det_idx] ^ (1LL << spawn_orbs[walker_idx][0]) ^ (1LL << spawn_orbs[walker_idx][1]);
                //                spawn_walker = -sing_det_parity(&new_det, spawn_orbs[walker_idx]) * walk_sign;
                spawn_walker = -walk_sign;
                
                find_neighbors(new_det, hub_len, b_tab, n_elec, tmp_orbs);
                hash_val = hash_fxn_hh(n_elec, tmp_orbs, proc_scrambler);
                proc_idx = hash_val % n_procs;
                
                send_dets[proc_idx][n_spawn[proc_idx]] = new_det | ini_flag;
                send_vals[proc_idx][n_spawn[proc_idx]] = spawn_walker;
                n_spawn[proc_idx]++;
            }
            
            // Death/cloning step
            matr_el = (1 - eps * (sol_mel[det_idx] - en_shift - hf_en)) * walk_sign;
            new_val = round_binomially(matr_el, n_walk, rngen_ptr);
            if (new_val == 0) {
                hash_val = hash_fxn_hh(n_elec, neighb_orbs[det_idx], det_hash->scrambler);
                del_ht(det_hash, sol_dets[det_idx], hash_val);
                push(det_stack, det_idx);
            }
            sol_vals[det_idx] = new_val;
        }
        // Communication
#ifdef USE_MPI
        MPI_Alltoall(n_spawn, 1, MPI_INT, recv_cts, 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Alltoallv(send_dets, n_spawn, displacements, MPI_LONG_LONG, recv_dets, recv_cts, displacements, MPI_LONG_LONG, MPI_COMM_WORLD);
        MPI_Alltoallv(send_vals, n_spawn, displacements, MPI_INT, recv_vals, recv_cts, displacements, MPI_INT, MPI_COMM_WORLD);
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
                find_neighbors(new_det, hub_len, b_tab, n_elec, tmp_orbs);
                hash_val = hash_fxn_hh(n_elec, tmp_orbs, det_hash->scrambler);
                idx_ptr = read_ht(det_hash, new_det, hash_val, !(!ini_flag));
                if (idx_ptr && *idx_ptr == -1) {
                    *idx_ptr = pop(det_stack);
                    if (*idx_ptr == -1) {
                        if (n_dets >= max_n_dets) {
                            fprintf(stderr, "Exceeded maximum length of determinant arrays\n");
                            return -1;
                        }
                        *idx_ptr = n_dets;
                        n_dets++;
                    }
                    sol_vals[*idx_ptr] = 0;
                    // copy neighbor lists over
                    for (det_idx = 0; det_idx < 2; det_idx++) {
                        for (n_walk = 0; n_walk < (1 + tmp_orbs[det_idx][0]); n_walk++) {
                            neighb_orbs[*idx_ptr][det_idx][n_walk] = tmp_orbs[det_idx][n_walk];
                        }
                    }
                    sol_mel[*idx_ptr] = hub_u * hub_diag(new_det, hub_len, b_tab);
                    sol_dets[*idx_ptr] = new_det;
                }
                if (ini_flag || (idx_ptr && (sol_vals[*idx_ptr] * recv_vals[proc_idx][walker_idx]) > 0)) {
                    sol_vals[*idx_ptr] += recv_vals[proc_idx][walker_idx];
                    if (sol_vals[*idx_ptr] == 0) {
                        push(det_stack, *idx_ptr);
                        del_ht(det_hash, new_det, hash_val);
                    }
                }
            }
        }
        // Communication
        if ((iterat + 1) % shift_interval == 0) {
            loc_norm = 0;
            for (det_idx = 0; det_idx < n_dets; det_idx++) {
                loc_norm += abs(sol_vals[det_idx]);
            }
            sum_mpi_i(loc_norm, &glob_norm, proc_rank, n_procs);
            adjust_shift(&en_shift, glob_norm, &last_norm, target_norm, shift_damping / eps / shift_interval);
            sum_mpi_i((int)n_nonz, &glob_nnonz, proc_rank, n_procs);
            if (proc_rank == ref_proc) {
                fprintf(walk_file, "%u\n", glob_norm);
                fprintf(shift_file, "%lf\n", en_shift);
                fprintf(nonz_file, "%d\n", glob_nnonz);
            }
        }
        
        spawn_walker = calc_ref_ovlp(&sol_dets[1], &sol_vals[1], n_dets - 1, neel_det, b_tab);
        
#ifdef USE_MPI
        MPI_Gather(&spawn_walker, 1, MPI_INT, recv_vals[0], 1, MPI_INT, ref_proc, MPI_COMM_WORLD);
#else
        recv_vals[0][0] = spawn_walker;
#endif
        if (proc_rank == ref_proc) {
            matr_el = sol_mel[0] * sol_vals[0];
            for (proc_idx = 0; proc_idx < n_procs; proc_idx++) {
                matr_el += recv_vals[0][proc_idx] * hub_t;
            }
            fprintf(num_file, "%lf\n", matr_el);
            fprintf(den_file, "%d\n", sol_vals[0]);
            printf("%6u, n walk: %7u, en est: %lf, shift: %lf, n_neel: %d\n", iterat, glob_norm, matr_el / sol_vals[0], en_shift, sol_vals[0]);
        }
        if ((iterat + 1) % save_interval == 0) {
            save_vec(result_dir, sol_dets, sol_vals, n_dets, sizeof(int));
            if (proc_rank == ref_proc) {
                fflush(num_file);
                fflush(den_file);
                fflush(shift_file);
                fflush(nonz_file);
            }
        }
    }
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}
