/*! \file
 *
 * \brief Implementation of the FCIQMC algorithm described in Booth et al. (2009)
 * for the Hubbard Model
 *
 * The steps involved in each iteration of an FCIQMC calculation are:
 * - Compress Hamiltonian matrix multinomially
 * - Stochastically round Hamiltonian matrix elements to integers
 * - Multiply current iterate by the compressed Hamiltonian matrix, scaled and
 * shifted to ensure convergence to the ground state
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <FRIES/io_utils.hpp>
#include <FRIES/Ext_Libs/dcmt/dc.h>
#include <FRIES/compress_utils.hpp>
#include <FRIES/Ext_Libs/argparse.h>
#define max_iter 1000000

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
    const char *ini_path = NULL;
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
        OPT_STRING('l', "load_dir", &load_dir, "Directory from which to load checkpoint files from a previous fciqmc calculation (in binary format, see documentation for save_vec())."),
        OPT_STRING('n', "ini_vec", &ini_path, "Prefix for files containing the vector with which to initialize the calculation (files must have names <ini_vec>dets and <ini_vec>vals and be text files)."),
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
    unsigned int spawn_length = target_walkers / n_procs / n_procs;
    DistVec<int> sol_vec(max_n_dets, spawn_length, rngen_ptr, n_orb, n_elec, hub_len, n_procs, INT);
    sol_vec.proc_scrambler_ = proc_scrambler;
    
    long long neel_det = gen_neel_det_1D(n_orb, n_elec, hub_dim);
    ref_proc = sol_vec.idx_to_proc(neel_det);
    size_t walker_idx;
    
    // Initialize solution vector
    if (load_dir) {
        sol_vec.load(load_dir);
    }
    else if (ini_path) {
        long long *load_dets = sol_vec.indices();
        int *load_vals = sol_vec[0];
        
        size_t n_dets = load_vec_txt(ini_path, load_dets, load_vals, INT);
        
        for (det_idx = 0; det_idx < n_dets; det_idx++) {
            sol_vec.add(load_dets[det_idx], load_vals[det_idx], ini_bit);
        }
    }
    else {
        if (ref_proc == proc_rank) {
            sol_vec.add(neel_det, 100, ini_bit);
        }
    }
    sol_vec.perform_add(ini_bit);
    loc_norm = sol_vec.local_norm();
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
        if (!num_file) {
            fprintf(stderr, "Could not open file for writing in directory %s\n", result_dir);
        }
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
        else if (ini_path) {
            fprintf(param_f, "Initializing calculation from vector at path %s\n", ini_path);
        }
        else {
            fprintf(param_f, "Initializing calculation from HF unit vector\n");
        }
        fclose(param_f);
    }
    
    unsigned int max_spawn = 500000; // should scale as max expected # on one determinant
    unsigned char (*spawn_orbs)[2] = (unsigned char (*)[2])malloc(sizeof(unsigned char) * 2 * max_spawn);
    
    long long ini_flag;
    unsigned int n_walk, n_success;
    int spawn_walker, walk_sign, new_val;
    long long new_det;
    double matr_el;
    double recv_nums[n_procs];
    
    unsigned int iterat;
    int glob_nnonz;
    int n_nonz;
    for (iterat = 0; iterat < max_iter; iterat++) {
        n_nonz = 0;
        for (det_idx = 0; det_idx < sol_vec.curr_size(); det_idx++) {
            int *curr_el = sol_vec[det_idx];
            long long curr_det = sol_vec.indices()[det_idx];
            n_walk = abs(*curr_el);
            if (n_walk == 0) {
                continue;
            }
            n_nonz++;
            ini_flag = n_walk > init_thresh;
            ini_flag <<= 2 * n_orb;
            walk_sign = 1 - ((*curr_el >> (sizeof(int) * 8 - 1)) & 2);
            
            // spawning step
            const Matrix<unsigned char> &neighb_orbs = sol_vec.neighb();
            matr_el = eps * hub_t * (neighb_orbs(det_idx, 0) + neighb_orbs(det_idx, n_elec + 1));
            n_success = round_binomially(matr_el, n_walk, rngen_ptr);
            
            if (n_success > max_spawn) {
                printf("Allocating more memory for spawning\n");
                max_spawn *= 2;
                spawn_orbs = (unsigned char (*)[2])realloc(spawn_orbs, sizeof(unsigned char) * 2 * max_spawn);
            }
            
            hub_multin(curr_det, n_elec, neighb_orbs[det_idx], n_success, rngen_ptr, spawn_orbs);
            
            for (walker_idx = 0; walker_idx < n_success; walker_idx++) {
                new_det = curr_det ^ (1LL << spawn_orbs[walker_idx][0]) ^ (1LL << spawn_orbs[walker_idx][1]);
                spawn_walker = -walk_sign;
                sol_vec.add(new_det, spawn_walker, ini_flag);
            }
            
            // Death/cloning step
            double *diag_el = sol_vec.matr_el_at_pos(det_idx);
            if (isnan(*diag_el)) {
                *diag_el = hub_diag(curr_det, hub_len, sol_vec.tabl()) * hub_u;
            }
            matr_el = (1 - eps * (*diag_el - en_shift - hf_en)) * walk_sign;
            new_val = round_binomially(matr_el, n_walk, rngen_ptr);
            if (new_val == 0 && sol_vec.indices()[det_idx] != neel_det) {
                sol_vec.del_at_pos(det_idx);
            }
            *curr_el = new_val;
        }
        sol_vec.perform_add(ini_bit);
        
        // Adjust shift
        if ((iterat + 1) % shift_interval == 0) {
            loc_norm = sol_vec.local_norm();
            sum_mpi_d(loc_norm, &glob_norm, proc_rank, n_procs);
            adjust_shift(&en_shift, glob_norm, &last_norm, target_norm, shift_damping / eps / shift_interval);
            sum_mpi_i((int)n_nonz, &glob_nnonz, proc_rank, n_procs);
            if (proc_rank == ref_proc) {
                fprintf(walk_file, "%u\n", (unsigned int) glob_norm);
                fprintf(shift_file, "%lf\n", en_shift);
                fprintf(nonz_file, "%d\n", glob_nnonz);
            }
        }
        
        // Calculate energy estimate
        matr_el = calc_ref_ovlp(sol_vec.indices(), sol_vec[0], sol_vec.curr_size(), neel_det, sol_vec.tabl(), sol_vec.type());
#ifdef USE_MPI
        MPI_Gather(&matr_el, 1, MPI_DOUBLE, recv_nums, 1, MPI_DOUBLE, ref_proc, MPI_COMM_WORLD);
#else
        recv_nums[0] = matr_el;
#endif
        if (proc_rank == ref_proc) {
            double *diag_el = sol_vec.matr_el_at_pos(0);
            if (isnan(*diag_el)) {
                *diag_el = hub_diag(neel_det, hub_len, sol_vec.tabl()) * hub_u;
            }
            int ref_element = sol_vec[0][0];
            matr_el = *diag_el * ref_element;
            for (proc_idx = 0; proc_idx < n_procs; proc_idx++) {
                matr_el += recv_nums[proc_idx] * hub_t;
            }
            fprintf(num_file, "%lf\n", matr_el);
            fprintf(den_file, "%d\n", ref_element);
            printf("%6u, n walk: %7u, en est: %lf, shift: %lf, n_neel: %d\n", iterat, (unsigned int)glob_norm, matr_el / ref_element, en_shift, ref_element);
        }
        
        // Save vector snapshot to disk
        if ((iterat + 1) % save_interval == 0) {
            sol_vec.save(result_dir);
            if (proc_rank == ref_proc) {
                fflush(num_file);
                fflush(den_file);
                fflush(shift_file);
                fflush(nonz_file);
            }
        }
    }
    sol_vec.save(result_dir);
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
