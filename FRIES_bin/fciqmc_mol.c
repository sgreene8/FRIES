/*! \file
 *
 * \brief Implementation of the FCIQMC algorithm described in Booth et al. (2009)
 * for a molecular system
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
#include <FRIES/Hamiltonians/near_uniform.h>
#include <FRIES/io_utils.h>
#include <FRIES/compress_utils.h>
#include <FRIES/Ext_Libs/argparse.h>
#include <FRIES/Hamiltonians/heat_bathPP.h>
#include <FRIES/Hamiltonians/molecule.h>
#define max_iter 1000000

static const char *const usage[] = {
    "fciqmc_mol [options] [[--] args]",
    "fciqmc_mol [options]",
    NULL,
};

typedef enum {
    near_uni,
    heat_bath
} h_dist;

int main(int argc, const char * argv[]) {
    const char *hf_path = NULL;
    const char *dist_str = NULL;
    const char *result_dir = "./";
    const char *load_dir = NULL;
    const char *ini_dir = NULL;
    unsigned int target_walkers = 0;
    unsigned int max_n_dets = 0;
    unsigned int init_thresh = 0;
    struct argparse_option options[] = {
        OPT_HELP(),
        OPT_STRING('d', "hf_path", &hf_path, "Path to the directory that contains the HF output files eris.txt, hcore.txt, symm.txt, hf_en.txt, and sys_params.txt"),
        OPT_INTEGER('t', "target", &target_walkers, "Target number of walkers, must be greater than the plateau value for this system"),
        OPT_STRING('q', "distribution", &dist_str, "Distribution to use to compress the Hamiltonian, either near-uniform (NU) or heat-bath Power-Pitzer (HB)"),
        OPT_STRING('y', "result_dir", &result_dir, "Directory in which to save output files"),
        OPT_INTEGER('p', "max_dets", &max_n_dets, "Maximum number of determinants on a single MPI process."),
        OPT_INTEGER('i', "initiator", &init_thresh, "Number of walkers on a determinant required to make it an initiator."),
        OPT_STRING('l', "load_dir", &load_dir, "Directory from which to load checkpoint files from a previous fciqmc calculation (in binary format, see documentation for save_vec())."),
        OPT_STRING('n', "ini_dir", &ini_dir, "Directory from which to read the initial vector for a new calculation (vector elements must be integers, see documentation for load_vec_txt())."),
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
    if (target_walkers == 0) {
        fprintf(stderr, "Error: target number of walkers not specified\n");
        return 0;
    }
    double target_norm = target_walkers;
    if (max_n_dets == 0) {
        fprintf(stderr, "Error: maximum number of determinants expected on each processor not specified.\n");
        return 0;
    }
    h_dist qmc_dist;
    if (strcmp(dist_str, "NU") == 0) {
        qmc_dist = near_uni;
    }
    else if (strcmp(dist_str, "HB") == 0) {
        qmc_dist = heat_bath;
    }
    else {
        fprintf(stderr, "Error: specified distribution for compressing Hamiltonian (%s) is not supported.\n", dist_str);
        return 0;
    }
    
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
    double hf_en = in_data.hf_en;
    
    unsigned int n_elec_unf = n_elec - n_frz;
    unsigned int tot_orb = n_orb + n_frz / 2;
    long long ini_bit = 1LL << (2 * n_orb);
    
    unsigned char *symm = in_data.symm;
    double (* h_core)[tot_orb] = (double (*)[tot_orb])in_data.hcore;
    double (* eris)[tot_orb][tot_orb][tot_orb] = (double (*)[tot_orb][tot_orb][tot_orb])in_data.eris;
    
    // Rn generator
    mt_struct *rngen_ptr = get_mt_parameter_id_st(32, 521, proc_rank, (unsigned int) time(NULL));
    sgenrand_mt((uint32_t) time(NULL), rngen_ptr);
    
    // Solution vector
    unsigned int spawn_length = target_walkers / n_procs / n_procs * 2;
    dist_vec *sol_vec = init_vec(max_n_dets, spawn_length, rngen_ptr, n_orb, n_elec_unf, INT, 0);
    size_t det_idx;
    
    unsigned char symm_lookup[n_irreps][n_orb + 1];
    gen_symm_lookup(symm, n_orb, n_irreps, symm_lookup);
    unsigned int unocc_symm_cts[n_irreps][2];
    
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
    sol_vec->proc_scrambler = proc_scrambler;
    
    long long hf_det = gen_hf_bitstring(n_orb, n_elec - n_frz);
    hf_proc = idx_to_proc(sol_vec, hf_det);
    
    unsigned char tmp_orbs[n_elec_unf];
    
    // Calculate HF column vector for energy estimator, and count # single/double excitations
    gen_orb_list(hf_det, sol_vec->tabl, tmp_orbs);
    size_t max_num_doub = count_doub_nosymm(n_elec_unf, n_orb);
    long long *hf_dets = malloc(max_num_doub * sizeof(long long));
    double *hf_mel = malloc(max_num_doub * sizeof(double));
    size_t n_hf_doub = gen_hf_ex(hf_det, tmp_orbs, n_elec_unf, tot_orb, symm, eris, n_frz, hf_dets, hf_mel);
    size_t n_hf_sing = count_singex(hf_det, tmp_orbs, symm, n_orb, symm_lookup, n_elec_unf);
    double p_doub = (double) n_hf_doub / (n_hf_sing + n_hf_doub);
    
    unsigned long long hf_hashes[n_hf_doub];
    for (det_idx = 0; det_idx < n_hf_doub; det_idx++) {
        hf_hashes[det_idx] = idx_to_hash(sol_vec, hf_dets[det_idx]);
    }
    size_t walker_idx;
    
    // Initialize solution vector
    if (load_dir) {
        // from previous FCIQMC calculation
        load_vec(sol_vec, load_dir);
    }
    else if (ini_dir) {
        // from an initial vector in .txt format
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
        // from Hartree-Fock
        if (hf_proc == proc_rank) {
            add_int(sol_vec, hf_det, 100, ini_bit);
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
    if (proc_rank == hf_proc) {
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
        
        // Describe parameters of this calculation
        strcpy(file_path, result_dir);
        strcat(file_path, "params.txt");
        FILE *param_f = fopen(file_path, "w");
        fprintf(param_f, "FCIQMC calculation\nHF path: %s\nepsilon (imaginary time step): %lf\nTarget number of walkers %u\nInitiator threshold: %u\n", hf_path, eps, target_walkers, init_thresh);
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
    unsigned char *spawn_orbs = malloc(sizeof(unsigned char) * 4 * max_spawn);
    double *spawn_probs = malloc(sizeof(double) * max_spawn);
    unsigned char (*sing_orbs)[2];
    unsigned char (*doub_orbs)[4];
    
    hb_info *hb_probs = NULL;
    if (qmc_dist == heat_bath) {
        hb_probs = set_up(tot_orb, n_orb, eris);
    }
    
    long long ini_flag;
    unsigned int n_walk, n_doub, n_sing;
    int spawn_walker, walk_sign, new_val;
    long long new_det;
    double matr_el;
    double recv_nums[n_procs];
    
    unsigned int iterat;
    int glob_nnonz;
    int n_nonz;
    for (iterat = 0; iterat < max_iter; iterat++) {
        n_nonz = 0;
        for (det_idx = 0; det_idx < sol_vec->curr_size; det_idx++) {
            int *curr_el = int_at_pos(sol_vec, det_idx);
            long long curr_det = sol_vec->indices[det_idx];
            n_walk = abs(*curr_el);
            if (n_walk == 0) {
                continue;
            }
            n_nonz++;
            ini_flag = n_walk > init_thresh;
            ini_flag <<= 2 * n_orb;
            walk_sign = 1 - ((*curr_el >> (sizeof(int) * 8 - 1)) & 2);
            
            // spawning step
            unsigned char *occ_orbs = orbs_at_pos(sol_vec, det_idx);
            count_symm_virt(unocc_symm_cts, occ_orbs, n_elec_unf,
                            n_orb, n_irreps, symm_lookup, symm);
            n_doub = bin_sample(n_walk, p_doub, rngen_ptr);
            n_sing = n_walk - n_doub;
            
            if (n_doub > max_spawn) {
                printf("Allocating more memory for spawning\n");
                max_spawn = n_doub * 3 / 2;
                spawn_orbs = realloc(spawn_orbs, sizeof(unsigned char) * 4 * max_spawn);
                spawn_probs = realloc(spawn_probs, sizeof(double) * max_spawn);
            }
            
            if (n_sing / 2 > max_spawn) {
                printf("Allocating more memory for spawning\n");
                max_spawn = n_sing * 3;
                spawn_orbs = realloc(spawn_orbs, sizeof(unsigned char) * 4 * max_spawn);
                spawn_probs = realloc(spawn_probs, sizeof(double) * max_spawn);
            }
            
            doub_orbs = (unsigned char (*)[4]) spawn_orbs;
            
            if (qmc_dist == near_uni) {
                n_doub = doub_multin(curr_det, occ_orbs, n_elec_unf, symm, n_orb, symm_lookup, unocc_symm_cts, n_doub, rngen_ptr, doub_orbs, spawn_probs);
            }
            else if (qmc_dist == heat_bath) {
                n_doub = hb_doub_multi(curr_det, occ_orbs, n_elec_unf, symm, hb_probs, (unsigned char *)symm_lookup, n_doub, rngen_ptr, doub_orbs, spawn_probs);
            }
            
            for (walker_idx = 0; walker_idx < n_doub; walker_idx++) {
                matr_el = doub_matr_el_nosgn(doub_orbs[walker_idx], tot_orb, eris, n_frz);
                matr_el *= eps / spawn_probs[walker_idx] / p_doub;
                spawn_walker = round_binomially(matr_el, 1, rngen_ptr);
                
                if (spawn_walker != 0) {
                    new_det = curr_det;
                    spawn_walker *= -doub_det_parity(&new_det, doub_orbs[walker_idx]) * walk_sign;
                    add_int(sol_vec, new_det, spawn_walker, ini_flag);
                }
            }
            
            sing_orbs = (unsigned char (*)[2]) spawn_orbs;
            n_sing = sing_multin(curr_det, occ_orbs, n_elec_unf, symm, n_orb, symm_lookup, unocc_symm_cts, n_sing, rngen_ptr, sing_orbs, spawn_probs);
            
            for (walker_idx = 0; walker_idx < n_sing; walker_idx++) {
                matr_el = sing_matr_el_nosgn(sing_orbs[walker_idx], occ_orbs, tot_orb, eris, h_core, n_frz, n_elec_unf);
                matr_el *= eps / spawn_probs[walker_idx] / (1 - p_doub);
                spawn_walker = round_binomially(matr_el, 1, rngen_ptr);
                
                if (spawn_walker != 0) {
                    new_det = curr_det;
                    spawn_walker *= -sing_det_parity(&new_det, sing_orbs[walker_idx]) * walk_sign;
                    add_int(sol_vec, new_det, spawn_walker, ini_flag);
                }
            }
            
            // Death/cloning step
            double *diag_el = &(sol_vec->matr_el[det_idx]);
            if (isnan(*diag_el)) {
                *diag_el = diag_matrel(occ_orbs, tot_orb, eris, h_core, n_frz, n_elec) - hf_en;
            }
            matr_el = (1 - eps * (*diag_el - en_shift)) * walk_sign;
            new_val = round_binomially(matr_el, n_walk, rngen_ptr);
            if (new_val == 0 && sol_vec->indices[det_idx] != hf_det) {
                del_at_pos(sol_vec, det_idx);
            }
            *curr_el = new_val;
        }
        perform_add(sol_vec, ini_bit);
        
        // Adjust shift
        if ((iterat + 1) % shift_interval == 0) {
            loc_norm = local_norm(sol_vec);
            sum_mpi_d(loc_norm, &glob_norm, proc_rank, n_procs);
            adjust_shift(&en_shift, glob_norm, &last_norm, target_norm, shift_damping / eps / shift_interval);
            sum_mpi_i((int)n_nonz, &glob_nnonz, proc_rank, n_procs);
            if (proc_rank == hf_proc) {
                fprintf(walk_file, "%u\n", (unsigned int) glob_norm);
                fprintf(shift_file, "%lf\n", en_shift);
                fprintf(nonz_file, "%d\n", glob_nnonz);
            }
        }
        
        // Calculate energy estimate
        matr_el = vec_dot(sol_vec, hf_dets, hf_mel, n_hf_doub, hf_hashes);
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
            int ref_element = ((int *)sol_vec->values)[0];
            fprintf(den_file, "%d\n", ref_element);
            printf("%6u, n walk: %7u, en est: %lf, shift: %lf\n", iterat, (unsigned int)glob_norm, matr_el / ref_element, en_shift);
        }
        
        // Save vector snapshot to disk
        if ((iterat + 1) % save_interval == 0) {
            save_vec(sol_vec, result_dir);
            if (proc_rank == hf_proc) {
                fflush(num_file);
                fflush(den_file);
                fflush(shift_file);
                fflush(nonz_file);
            }
        }
    }
    save_vec(sol_vec, result_dir);
    if (proc_rank == hf_proc) {
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

