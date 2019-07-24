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
#define max_iter 100000

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
    
    // Rn generator
    mt_struct *rngen_ptr = get_mt_parameter_id_st(32, 521, proc_rank, (unsigned int) time(NULL));
    sgenrand_mt((uint32_t) time(NULL), rngen_ptr);
    size_t det_idx;
    
    // Initialize hash function for processors and vector
    unsigned int proc_scrambler[2 * n_orb];
    double loc_norm, glob_norm;
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
    
    // Solution vector
    unsigned int spawn_length = 100000;
    dist_vec *sol_vec = init_vec(max_n_dets, spawn_length, rngen_ptr, n_orb, n_elec, DOUB, hub_len);
    sol_vec->proc_scrambler = proc_scrambler;
    
    long long neel_det = gen_hub_bitstring(n_orb, n_elec, hub_dim);
    ref_proc = idx_to_proc(sol_vec, neel_det);
    size_t walker_idx;
    
    // Initialize solution vector
    if (load_dir) {
        load_vec(sol_vec, load_dir);
    }
    else if (ini_dir) {
        long long *load_dets = sol_vec->indices;
        int *load_vals = sol_vec->values;
        
        char buf[100];
        sprintf(buf, "%sqmc_", ini_dir);
        size_t n_dets = load_vec_txt(buf, load_dets, load_vals, INT);
        
        for (det_idx = 0; det_idx < n_dets; det_idx++) {
            add_int(sol_vec, load_dets[det_idx], load_vals[det_idx], ini_bit);
        }
    }
    else {
        if (ref_proc == proc_rank) {
            add_doub(sol_vec, neel_det, 1, ini_bit);
        }
    }
    perform_add(sol_vec, ini_bit);
    loc_norm = local_norm(sol_vec);
    sum_mpi_d(loc_norm, &glob_norm, proc_rank, n_procs);
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
        //        FILE *param_f = fopen(file_path, "w");
        //        fprintf(param_f, "FCIQMC calculation\nHubbard-Holstein parameters path: %s\nepsilon (imaginary time step): %lf\nTarget number of walkers %u\nInitiator threshold: %u\n", params_path, eps, target_walkers, init_thresh);
        //        if (load_dir) {
        //            fprintf(param_f, "Restarting calculation from %s\n", load_dir);
        //        }
        //        else if (ini_dir) {
        //            fprintf(param_f, "Initializing calculation from vector at path %s\n", ini_dir);
        //        }
        //        else {
        //            fprintf(param_f, "Initializing calculation from HF unit vector\n");
        //        }
        //        fclose(param_f);
    }
    
    unsigned int max_spawn = 500000; // should scale as max expected # on one determinant
    unsigned char (*spawn_orbs)[2] = malloc(sizeof(unsigned char) * 2 * max_spawn);
    
    long long ini_flag;
    unsigned int n_success;
    double new_val;
    long long new_det;
    double matr_el;
    double recv_nums[n_procs];
    
    unsigned int iterat;
    int glob_nnonz;
    int n_nonz;
    for (iterat = 0; iterat < max_iter; iterat++) {
        n_nonz = 0;
        for (det_idx = 0; det_idx < sol_vec->curr_size; det_idx++) {
            double *curr_el = doub_at_pos(sol_vec, det_idx);
            long long curr_det = sol_vec->indices[det_idx];
            if (*curr_el == 0) {
                continue;
            }
            n_nonz++;
            ini_flag = 1;
            ini_flag <<= 2 * n_orb;
            
            // spawning step
            unsigned char (*neighb_orbs)[2][n_elec + 1] = (unsigned char (*)[2][n_elec + 1])sol_vec->neighb;
            n_success = hub_all(curr_det, n_elec, neighb_orbs[det_idx], spawn_orbs);
            
            for (walker_idx = 0; walker_idx < n_success; walker_idx++) {
                new_det = curr_det ^ (1LL << spawn_orbs[walker_idx][0]) ^ (1LL << spawn_orbs[walker_idx][1]);
                add_doub(sol_vec, new_det, eps * hub_t * (*curr_el), ini_flag);
            }
            
            // Death/cloning step
            double *diag_el = &(sol_vec->matr_el[det_idx]);
            if (isnan(*diag_el)) {
                *diag_el = hub_diag(curr_det, hub_len, sol_vec->tabl) * hub_u;
            }
            matr_el = (1 - eps * (*diag_el - en_shift - hf_en));
            new_val = matr_el * (*curr_el);
            if (new_val == 0 && sol_vec->indices[det_idx] != neel_det) {
                del_at_pos(sol_vec, det_idx);
            }
            *curr_el = new_val;
        }
        perform_add(sol_vec, ini_bit);
        
        // Communication
        if ((iterat + 1) % shift_interval == 0) {
            loc_norm = local_norm(sol_vec);
            sum_mpi_d(loc_norm, &glob_norm, proc_rank, n_procs);
            adjust_shift(&en_shift, glob_norm, &last_norm, target_norm, shift_damping / eps / shift_interval);
            sum_mpi_i((int)n_nonz, &glob_nnonz, proc_rank, n_procs);
            if (proc_rank == ref_proc) {
                //                fprintf(walk_file, "%u\n", (unsigned int) glob_norm);
                //                fprintf(shift_file, "%lf\n", en_shift);
                //                fprintf(nonz_file, "%d\n", glob_nnonz);
            }
        }
        new_val = calc_ref_ovlp(sol_vec->indices, sol_vec->values, sol_vec->curr_size, neel_det, sol_vec->tabl, sol_vec->type);
        
#ifdef USE_MPI
        MPI_Gather(&new_val, 1, MPI_INT, recv_nums, 1, MPI_INT, ref_proc, MPI_COMM_WORLD);
#else
        recv_nums[0] = new_val;
#endif
        if (proc_rank == ref_proc) {
            double *diag_el = &(sol_vec->matr_el[0]);
            if (isnan(*diag_el)) {
                *diag_el = hub_diag(neel_det, hub_len, sol_vec->tabl) * hub_u;
            }
            double ref_element = ((double *)sol_vec->values)[0];
            matr_el = *diag_el * ref_element;
            for (proc_idx = 0; proc_idx < n_procs; proc_idx++) {
                matr_el += recv_nums[proc_idx] * -hub_t;
            }
            //            fprintf(num_file, "%lf\n", matr_el);
            //            fprintf(den_file, "%d\n", ref_element);
            printf("%6u, en est: %lf, shift: %lf, n_neel: %lf\n", iterat, matr_el / ref_element, en_shift, ref_element);
        }
        if ((iterat + 1) % save_interval == 0) {
            save_vec(sol_vec, result_dir);
            if (proc_rank == ref_proc) {
                fflush(num_file);
                fflush(den_file);
                fflush(shift_file);
                fflush(nonz_file);
            }
        }
    }
    save_vec(sol_vec, result_dir);
    if (proc_rank == ref_proc) {
        fclose(num_file);
        fclose(den_file);
        fclose(shift_file);
        fclose(nonz_file);
    }
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}

