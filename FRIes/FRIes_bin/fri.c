#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "../FRIes/Hamiltonians/near_uniform.h"
#include "../FRIes/io_utils.h"
#include "../FRIes/Ext_Libs/dc.h"
#include "../FRIes/compress_utils.h"
#include "../FRIes/Ext_Libs/argparse.h"
#define max_iter 10000000

static const char *const usage[] = {
    "fri [options] [[--] args]",
    "fri [options]",
    NULL,
};


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
    
    unsigned char *symm = in_data.symm;
    double (* h_core)[tot_orb] = (double (*)[tot_orb])in_data.hcore;
    double (* eris)[tot_orb][tot_orb][tot_orb] = (double (*)[tot_orb][tot_orb][tot_orb])in_data.eris;
    
    // Rn generator
    mt_struct *rngen_ptr = get_mt_parameter_id_st(32, 607, proc_rank, (unsigned int) time(NULL));
    sgenrand_mt((uint32_t) time(NULL), rngen_ptr);
    
    // Solution vector
    unsigned int spawn_length = matr_samp * 2 / n_procs / n_procs;
    dist_vec *sol_vec = init_vec(max_n_dets, spawn_length, rngen_ptr, n_orb, n_elec_unf, DOUB, 0);
    size_t det_idx;
    
    unsigned char symm_lookup[n_irreps][n_orb + 1];
    gen_symm_lookup(symm, n_orb, n_irreps, symm_lookup);
    unsigned int unocc_symm_cts[n_irreps][2];
    
    // Initialize hash function for processors and vector
    unsigned int proc_scrambler[2 * n_orb];
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
    
    // Initialize solution vector
    if (load_dir) {
        load_vec(sol_vec, load_dir);
    }
    else if (ini_dir) {
        long long *load_dets = sol_vec->indices;
        double *load_vals = sol_vec->values;
        
        char buf[100];
        sprintf(buf, "%sfri_", ini_dir);
        long long n_dets = load_vec_txt(buf, load_dets, load_vals, DOUB);
        
        for (det_idx = 0; det_idx < n_dets; det_idx++) {
            add_doub(sol_vec, load_dets[det_idx], load_vals[det_idx], ini_bit);
        }
    }
    else {
        if (hf_proc == proc_rank) {
            add_doub(sol_vec, hf_det, 1, ini_bit);
        }
    }
    perform_add(sol_vec, ini_bit);
    loc_norms[proc_rank] = local_norm(sol_vec);
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
    
    unsigned int max_spawn = matr_samp; // should scale as max expected # from one determinant
    unsigned char *spawn_orbs = malloc(sizeof(unsigned char) * 4 * max_spawn);
    double *spawn_probs = malloc(sizeof(double) * max_spawn);
    unsigned char (*sing_orbs)[2];
    unsigned char (*doub_orbs)[4];
    
    long long ini_flag;
    unsigned int n_walk, n_doub, n_sing;
    double last_one_norm = 0;
    long long new_det;
    double matr_el;
    double recv_nums[n_procs];
    
    // Variables for compression
    double rn_sys = 0;
    double lbound;
    double weight;
    int glob_n_nonz; // Number of nonzero elements in whole vector (across all processors)
    size_t *srt_arr = malloc(sizeof(size_t) * max_n_dets);
    int *keep_exact = calloc(max_n_dets, sizeof(int));
    
    unsigned int iterat;
    for (iterat = 0; iterat < max_iter; iterat++) {
        sum_mpi_i(sol_vec->n_nonz, &glob_n_nonz, proc_rank, n_procs);
        
        // Systematic sampling to determine number of samples for each column
        if (proc_rank == 0) {
            rn_sys = genrand_mt(rngen_ptr) / (1. + UINT32_MAX);
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
        for (det_idx = 0; det_idx < sol_vec->curr_size; det_idx++) {
            double *curr_el = doub_at_pos(sol_vec, det_idx);
            long long curr_det = sol_vec->indices[det_idx];
            weight = fabs(*curr_el);
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
            
            // spawning step
            unsigned char *occ_orbs = orbs_at_pos(sol_vec, det_idx);
            count_symm_virt(unocc_symm_cts, occ_orbs, n_elec_unf,
                            n_orb, n_irreps, symm_lookup, symm);
            n_doub = bin_sample(n_walk, p_doub, rngen_ptr);
            n_sing = n_walk - n_doub;
            
            if (n_doub > max_spawn || n_sing / 2 > max_spawn) {
                printf("Allocating more memory for spawning\n");
                max_spawn *= 2;
                spawn_orbs = realloc(spawn_orbs, sizeof(unsigned char) * 4 * max_spawn);
                spawn_probs = realloc(spawn_probs, sizeof(double) * max_spawn);
            }
            doub_orbs = (unsigned char (*)[4]) spawn_orbs;
            n_doub = doub_multin(curr_det, occ_orbs, n_elec_unf, symm, n_orb, symm_lookup, unocc_symm_cts, n_doub, rngen_ptr, doub_orbs, spawn_probs);
            
            size_t walker_idx;
            for (walker_idx = 0; walker_idx < n_doub; walker_idx++) {
                matr_el = doub_matr_el_nosgn(doub_orbs[walker_idx], tot_orb, eris, n_frz);
                if (fabs(matr_el) > 1e-9) {
                    new_det = curr_det;
                    matr_el *= -eps / spawn_probs[walker_idx] / p_doub / n_walk * (*curr_el) * doub_det_parity(&new_det, doub_orbs[walker_idx]);
                    add_doub(sol_vec, new_det, matr_el, ini_flag);
                }
            }
            
            sing_orbs = (unsigned char (*)[2]) spawn_orbs;
            n_sing = sing_multin(curr_det, occ_orbs, n_elec_unf, symm, n_orb, symm_lookup, unocc_symm_cts, n_sing, rngen_ptr, sing_orbs, spawn_probs);
            
            for (walker_idx = 0; walker_idx < n_sing; walker_idx++) {
                matr_el = sing_matr_el_nosgn(sing_orbs[walker_idx], occ_orbs, tot_orb, eris, h_core, n_frz, n_elec_unf);
                if (fabs(matr_el) > 1e-9) {
                    new_det = curr_det;
                    matr_el *= -eps / spawn_probs[walker_idx] / (1 - p_doub) / n_walk * (*curr_el) * sing_det_parity(&new_det, sing_orbs[walker_idx]);
                    add_doub(sol_vec, new_det, matr_el, ini_flag);
                }
            }
            
            // Death/cloning step
            double *diag_el = &(sol_vec->matr_el[det_idx]);
            if (isnan(*diag_el)) {
                *diag_el = diag_matrel(occ_orbs, tot_orb, eris, h_core, n_frz, n_elec) - hf_en;
            }
            *curr_el *= 1 - eps * (*diag_el - en_shift);
        }
        perform_add(sol_vec, ini_bit);
        
        // Compression step
        unsigned int n_samp = target_nonz;
        loc_norms[proc_rank] = find_preserve((double *)sol_vec->values, srt_arr, keep_exact, sol_vec->curr_size, &n_samp, &glob_norm);
        
        // Adjust shift
        if ((iterat + 1) % shift_interval == 0) {
            adjust_shift(&en_shift, glob_norm, &last_one_norm, target_norm, shift_damping / shift_interval / eps);
            if (proc_rank == hf_proc) {
                fprintf(shift_file, "%lf\n", en_shift);
                fprintf(norm_file, "%lf\n", glob_norm);
            }
        }
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
            double ref_element = ((double *)sol_vec->values)[0];
            fprintf(den_file, "%lf\n", ref_element);
            printf("%6u, en est: %lf, shift: %lf, norm: %lf\n", iterat, matr_el / ref_element, en_shift, glob_norm);
        }
        
        if (proc_rank == 0) {
            rn_sys = genrand_mt(rngen_ptr) / (1. + UINT32_MAX);
        }
#ifdef USE_MPI
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, loc_norms, 1, MPI_DOUBLE, MPI_COMM_WORLD);
//        MPI_Bcast(&rn_sys, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
        sys_comp(sol_vec->values, sol_vec->curr_size, loc_norms, n_samp, keep_exact, rn_sys);
        for (det_idx = 0; det_idx < sol_vec->curr_size; det_idx++) {
            if (keep_exact[det_idx] && sol_vec->indices[det_idx] != hf_det) {
                del_at_pos(sol_vec, det_idx);
                keep_exact[det_idx] = 0;
            }
        }
        
        if ((iterat + 1) % save_interval == 0) {
            save_vec(sol_vec, result_dir);
            if (proc_rank == hf_proc) {
                fflush(num_file);
                fflush(den_file);
                fflush(shift_file);
            }
        }
    }
    save_vec(sol_vec, result_dir);
    if (proc_rank == hf_proc) {
        fclose(num_file);
        fclose(den_file);
        fclose(shift_file);
    }
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}
