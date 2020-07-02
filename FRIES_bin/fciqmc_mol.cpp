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
#include <FRIES/Hamiltonians/near_uniform.hpp>
#include <FRIES/io_utils.hpp>
#include <FRIES/compress_utils.hpp>
#include <FRIES/Ext_Libs/argparse.h>
#include <FRIES/Hamiltonians/heat_bathPP.hpp>
#include <FRIES/Hamiltonians/molecule.hpp>

static const char *const usage[] = {
    "fciqmc_mol [options] [[--] args]",
    "fciqmc_mol [options]",
    NULL,
};

using namespace std;

int main(int argc, const char * argv[]) {
    const char *hf_path = NULL;
    const char *dist_str = NULL;
    const char *result_dir = "./";
    const char *load_dir = NULL;
    const char *ini_path = NULL;
    const char *trial_path = NULL;
    unsigned int target_walkers = 0;
    unsigned int max_n_dets = 0;
    unsigned int init_thresh = 0;
    unsigned int max_iter = 1000000;
    struct argparse_option options[] = {
        OPT_HELP(),
        OPT_STRING('d', "hf_path", &hf_path, "Path to the directory that contains the HF output files eris.txt, hcore.txt, symm.txt, hf_en.txt, and sys_params.txt"),
        OPT_INTEGER('t', "target", &target_walkers, "Target number of walkers, must be greater than the plateau value for this system"),
        OPT_STRING('q', "distribution", &dist_str, "Distribution to use to compress the Hamiltonian, either near-uniform (NU) or heat-bath Power-Pitzer (HB)"),
        OPT_STRING('y', "result_dir", &result_dir, "Directory in which to save output files"),
        OPT_INTEGER('p', "max_dets", &max_n_dets, "Maximum number of determinants on a single MPI process."),
        OPT_INTEGER('i', "initiator", &init_thresh, "Number of walkers on a determinant required to make it an initiator."),
        OPT_STRING('l', "load_dir", &load_dir, "Directory from which to load checkpoint files from a previous fciqmc calculation (in binary format, see documentation for DistVec::save() and DistVec::load())."),
        OPT_STRING('n', "ini_vec", &ini_path, "Prefix for files containing the vector with which to initialize the calculation (files must have names <ini_vec>dets and <ini_vec>vals and be text files)."),
        OPT_STRING('v', "trial_vec", &trial_path, "Prefix for files containing the vector with which to calculate the energy (files must have names <trial_vec>dets and <trial_vec>vals and be text files)."),
        OPT_INTEGER('I', "max_iter", &max_iter, "Maximum number of iterations to run the calculation."),
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
    
    uint8_t *symm = in_data.symm;
    Matrix<double> *h_core = in_data.hcore;
    FourDArr *eris = in_data.eris;
    
    // Rn generator
    mt_struct *rngen_ptr = get_mt_parameter_id_st(32, 521, proc_rank, (unsigned int) time(NULL));
    sgenrand_mt((uint32_t) time(NULL), rngen_ptr);
    
    // Solution vector
    unsigned int spawn_length = target_walkers / n_procs / n_procs * 2;
    std::function<double(const uint8_t *)> diag_shortcut = [tot_orb, eris, h_core, n_frz, n_elec, hf_en](const uint8_t *occ_orbs) {
        return diag_matrel(occ_orbs, tot_orb, *eris, *h_core, n_frz, n_elec) - hf_en;
    };
    size_t det_size = CEILING(2 * n_orb, 8);
    size_t det_idx;
    
    Matrix<uint8_t> symm_lookup(n_irreps, n_orb + 1);
    gen_symm_lookup(symm, symm_lookup);
    unsigned int unocc_symm_cts[n_irreps][2];
    
    // Initialize hash function for processors and vector
    std::vector<uint32_t> proc_scrambler(2 * n_orb);
    double loc_norm, glob_norm;
    double last_norm = 0;
    
    if (load_dir) {
        load_proc_hash(load_dir, proc_scrambler.data());
    }
    else {
        if (proc_rank == 0) {
            for (det_idx = 0; det_idx < 2 * n_orb; det_idx++) {
                proc_scrambler[det_idx] = genrand_mt(rngen_ptr);
            }
            save_proc_hash(result_dir, proc_scrambler.data(), 2 * n_orb);
        }
#ifdef USE_MPI
        MPI_Bcast(proc_scrambler.data(), 2 * n_orb, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
#endif
    }
    std::vector<uint32_t> vec_scrambler(2 * n_orb);
    for (det_idx = 0; det_idx < 2 * n_orb; det_idx++) {
        vec_scrambler[det_idx] = genrand_mt(rngen_ptr);
    }
    
    DistVec<int> sol_vec(max_n_dets, spawn_length, n_orb * 2, n_elec_unf, n_procs, diag_shortcut, NULL, 1, proc_scrambler, vec_scrambler);
    
    uint8_t hf_det[det_size];
    gen_hf_bitstring(n_orb, n_elec - n_frz, hf_det);
    hf_proc = sol_vec.idx_to_proc(hf_det);
    
    uint8_t tmp_orbs[n_elec_unf];
    unsigned int max_spawn = 500000; // should scale as max expected # on one determinant
    uint8_t *spawn_orbs = (uint8_t *) malloc(sizeof(uint8_t) * 4 * max_spawn);
    double *spawn_probs = (double *) malloc(sizeof(double) * max_spawn);
    uint8_t (*sing_orbs)[2] = (uint8_t (*)[2]) spawn_orbs;
    uint8_t (*doub_orbs)[4] = (uint8_t (*)[4]) spawn_orbs;
    
# pragma mark Set up trial vector
    size_t n_trial;
    size_t n_ex = n_orb * n_orb * n_elec_unf * n_elec_unf;
    Matrix<uint8_t> &load_dets = sol_vec.indices();
    double *load_vals = (double *)sol_vec.values();
    if (trial_path) { // load trial vector from file
        n_trial = load_vec_txt(trial_path, load_dets, load_vals, DOUB);
    }
    else {
        n_trial = 1;
    }
    DistVec<double> trial_vec(n_trial, n_trial, n_orb * 2, n_elec_unf, n_procs, proc_scrambler, vec_scrambler);
    DistVec<double> htrial_vec(n_trial * n_ex / n_procs, n_trial * n_ex / n_procs, n_orb * 2, n_elec_unf, n_procs, diag_shortcut, NULL, 2, proc_scrambler, vec_scrambler);
    if (trial_path) { // load trial vector from file
        for (det_idx = 0; det_idx < n_trial; det_idx++) {
            trial_vec.add(load_dets[det_idx], load_vals[det_idx], 1);
            htrial_vec.add(load_dets[det_idx], load_vals[det_idx], 1);
        }
    }
    else { // Otherwise, use HF as trial vector
        trial_vec.add(hf_det, 1, 1);
        htrial_vec.add(hf_det, 1, 1);
    }
    trial_vec.perform_add();
    htrial_vec.perform_add();
    
    trial_vec.collect_procs();
    uintmax_t *trial_hashes = (uintmax_t *)malloc(sizeof(uintmax_t) * trial_vec.curr_size());
    for (det_idx = 0; det_idx < trial_vec.curr_size(); det_idx++) {
        trial_hashes[det_idx] = sol_vec.idx_to_hash(trial_vec.indices()[det_idx], tmp_orbs);
    }
    
    // Calculate H * trial vector, and accumulate results on each processor
    h_op_offdiag(htrial_vec, symm, tot_orb, *eris, *h_core, spawn_orbs, n_frz, n_elec_unf, 1, 1);
    htrial_vec.set_curr_vec_idx(0);
    h_op_diag(htrial_vec, 0, 0, 1);
    htrial_vec.add_vecs(0, 1);
    htrial_vec.collect_procs();
    uintmax_t *htrial_hashes = (uintmax_t *)malloc(sizeof(uintmax_t) * htrial_vec.curr_size());
    for (det_idx = 0; det_idx < htrial_vec.curr_size(); det_idx++) {
        htrial_hashes[det_idx] = sol_vec.idx_to_hash(htrial_vec.indices()[det_idx], tmp_orbs);
    }
    
    // Count # single/double excitations from HF
    sol_vec.gen_orb_list(hf_det, tmp_orbs);
    size_t n_hf_doub = doub_ex_symm(hf_det, tmp_orbs, n_elec_unf, n_orb, doub_orbs, symm);
    size_t n_hf_sing = count_singex(hf_det, tmp_orbs, symm, n_orb, symm_lookup, n_elec_unf);
    double p_doub = (double) n_hf_doub / (n_hf_sing + n_hf_doub);
    
    size_t walker_idx;
    char file_path[300];
    FILE *walk_file = NULL;
    FILE *num_file = NULL;
    FILE *den_file = NULL;
    FILE *shift_file = NULL;
    FILE *nonz_file = NULL;
    FILE *ini_file = NULL;
    
    // Initialize solution vector
    unsigned int max_vals = 0;
    if (load_dir) {
        // from previous FCIQMC calculation
        sol_vec.load(load_dir);
        
        // load energy shift (see https://stackoverflow.com/questions/13790662/c-read-only-last-line-of-a-file-no-loops)
        static const long max_len = 20;
        sprintf(file_path, "%sS.txt", load_dir);
        shift_file = fopen(file_path, "rb");
        fseek(shift_file, -max_len, SEEK_END);
        (void) fread(file_path, max_len, 1, shift_file);
        fclose(shift_file);
        shift_file = NULL;
        
        file_path[max_len - 1] = '\0';
        char *last_newline = strrchr(file_path, '\n');
        char *last_line = last_newline + 1;
        
        sscanf(last_line, "%lf", &en_shift);
    }
    else if (ini_path) {
        // from an initial vector in .txt format
        Matrix<uint8_t> &load_dets = sol_vec.indices();
        int *load_vals = sol_vec.values();
        
        size_t n_dets = load_vec_txt(ini_path, load_dets, load_vals, INT);
        
        for (det_idx = 0; det_idx < n_dets; det_idx++) {
            if (abs(load_vals[det_idx]) > max_vals) {
                max_vals = abs(load_vals[det_idx]);
            }
            sol_vec.add(load_dets[det_idx], load_vals[det_idx], 1);
        }
    }
    else {
        // from Hartree-Fock
        if (hf_proc == proc_rank) {
            sol_vec.add(hf_det, 100, 1);
        }
    }
    sol_vec.perform_add();
    loc_norm = sol_vec.local_norm();
    glob_norm = sum_mpi(loc_norm, proc_rank, n_procs);
    if (load_dir) {
        last_norm = glob_norm;
    }
    
    if (max_vals > spawn_length) {
        printf("Allocating more memory for spawning\n");
        max_spawn = max_vals * 1.2;
        spawn_orbs = (uint8_t *)realloc(spawn_orbs, sizeof(uint8_t) * 4 * max_spawn);
        sing_orbs = (uint8_t (*)[2]) spawn_orbs;
        doub_orbs = (uint8_t (*)[4]) spawn_orbs;
        spawn_probs = (double *) realloc(spawn_probs, sizeof(double) * max_spawn);
    }
    
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
        strcpy(file_path, result_dir);
        strcat(file_path, "nini.txt");
        ini_file = fopen(file_path, "a");
        
        // Describe parameters of this calculation
        strcpy(file_path, result_dir);
        strcat(file_path, "params.txt");
        FILE *param_f = fopen(file_path, "w");
        fprintf(param_f, "FCIQMC calculation\nHF path: %s\nepsilon (imaginary time step): %lf\nTarget number of walkers %u\nInitiator threshold: %u\n", hf_path, eps, target_walkers, init_thresh);
        if (load_dir) {
            fprintf(param_f, "Restarting calculation from %s\n", load_dir);
        }
        else if (ini_path) {
            fprintf(param_f, "Initializing calculation from vector files with prefix %s\n", ini_path);
        }
        else {
            fprintf(param_f, "Initializing calculation from HF unit vector\n");
        }
        fclose(param_f);
    }
    
    hb_info *hb_probs = NULL;
    if (qmc_dist == heat_bath) {
        hb_probs = set_up(tot_orb, n_orb, *eris);
    }
    
    int ini_flag;
    unsigned int n_walk, n_doub, n_sing;
    int spawn_walker, walk_sign, new_val;
    uint8_t new_det[det_size];
    double matr_el;
    double recv_nums[n_procs];
    double recv_dens[n_procs];
    
    unsigned int iterat;
    int glob_nnonz;
    int n_nonz;
    size_t n_ini;
    for (iterat = 0; iterat < max_iter; iterat++) {
        n_nonz = 0;
        n_ini = 0;
        for (det_idx = 0; det_idx < sol_vec.curr_size(); det_idx++) {
            int *curr_el = sol_vec[det_idx];
            uint8_t *curr_det = sol_vec.indices()[det_idx];
            n_walk = abs(*curr_el);
            if (n_walk == 0) {
                continue;
            }
            n_nonz++;
            ini_flag = n_walk > init_thresh;
            n_ini += ini_flag;
            walk_sign = 1 - ((*curr_el >> (sizeof(int) * 8 - 1)) & 2);
            
            // spawning step
            uint8_t *occ_orbs = sol_vec.orbs_at_pos(det_idx);
            count_symm_virt(unocc_symm_cts, occ_orbs, n_elec_unf,
                            n_orb, n_irreps, symm_lookup, symm);
            n_doub = bin_sample(n_walk, p_doub, rngen_ptr);
            n_sing = n_walk - n_doub;
            
            if (n_doub > max_spawn) {
                printf("Allocating more memory for spawning\n");
                max_spawn = n_doub * 3 / 2;
                spawn_orbs = (uint8_t *)realloc(spawn_orbs, sizeof(uint8_t) * 4 * max_spawn);
                spawn_probs = (double *) realloc(spawn_probs, sizeof(double) * max_spawn);
                sing_orbs = (uint8_t (*)[2]) spawn_orbs;
                doub_orbs = (uint8_t (*)[4]) spawn_orbs;
            }
            
            if (n_sing / 2 > max_spawn) {
                printf("Allocating more memory for spawning\n");
                max_spawn = n_sing * 3;
                spawn_orbs = (uint8_t *)realloc(spawn_orbs, sizeof(uint8_t) * 4 * max_spawn);
                sing_orbs = (uint8_t (*)[2]) spawn_orbs;
                doub_orbs = (uint8_t (*)[4]) spawn_orbs;
                spawn_probs = (double *) realloc(spawn_probs, sizeof(double) * max_spawn);
            }
            
            if (qmc_dist == near_uni) {
                n_doub = doub_multin(curr_det, occ_orbs, n_elec_unf, symm, n_orb, symm_lookup, unocc_symm_cts, n_doub, rngen_ptr, doub_orbs, spawn_probs);
            }
            else if (qmc_dist == heat_bath) {
                n_doub = hb_doub_multi(curr_det, occ_orbs, n_elec_unf, symm, hb_probs, symm_lookup, n_doub, rngen_ptr, doub_orbs, spawn_probs);
            }
            
            for (walker_idx = 0; walker_idx < n_doub; walker_idx++) {
                matr_el = doub_matr_el_nosgn(doub_orbs[walker_idx], tot_orb, *eris, n_frz);
                matr_el *= eps / spawn_probs[walker_idx] / p_doub;
                spawn_walker = round_binomially(matr_el, 1, rngen_ptr);
                
                if (spawn_walker != 0) {
                    memcpy(new_det, curr_det, det_size);
                    spawn_walker *= -doub_det_parity(new_det, doub_orbs[walker_idx]) * walk_sign;
                    sol_vec.add(new_det, spawn_walker, ini_flag);
                }
            }
            
            n_sing = sing_multin(curr_det, occ_orbs, n_elec_unf, symm, n_orb, symm_lookup, unocc_symm_cts, n_sing, rngen_ptr, sing_orbs, spawn_probs);
            
            for (walker_idx = 0; walker_idx < n_sing; walker_idx++) {
                matr_el = sing_matr_el_nosgn(sing_orbs[walker_idx], occ_orbs, tot_orb, *eris, *h_core, n_frz, n_elec_unf);
                matr_el *= eps / spawn_probs[walker_idx] / (1 - p_doub);
                spawn_walker = round_binomially(matr_el, 1, rngen_ptr);
                
                if (spawn_walker != 0) {
                    memcpy(new_det, curr_det, det_size);
                    spawn_walker *= -sing_det_parity(new_det, sing_orbs[walker_idx]) * walk_sign;
                    sol_vec.add(new_det, spawn_walker, ini_flag);
                }
            }
            
            // Death/cloning step
            double diag_el = sol_vec.matr_el_at_pos(det_idx);
            matr_el = (1 - eps * (diag_el - en_shift)) * walk_sign;
            new_val = round_binomially(matr_el, n_walk, rngen_ptr);
            if (new_val == 0 && sol_vec.indices()[det_idx] != hf_det) {
                sol_vec.del_at_pos(det_idx);
            }
            *curr_el = new_val;
        }
        sol_vec.perform_add();
        
        // Adjust shift
        if ((iterat + 1) % shift_interval == 0) {
            loc_norm = sol_vec.local_norm();
            glob_norm = sum_mpi(loc_norm, proc_rank, n_procs);
            adjust_shift(&en_shift, glob_norm, &last_norm, target_norm, shift_damping / eps / shift_interval);
            glob_nnonz = sum_mpi((int)n_nonz, proc_rank, n_procs);
            if (proc_rank == hf_proc) {
                fprintf(walk_file, "%u\n", (unsigned int) glob_norm);
                fprintf(shift_file, "%lf\n", en_shift);
                fprintf(nonz_file, "%d\n", glob_nnonz);
            }
        }
        
        // Calculate energy estimate
        matr_el = sol_vec.dot(htrial_vec.indices(), htrial_vec.values(), htrial_vec.curr_size(), htrial_hashes);
        double denom = sol_vec.dot(trial_vec.indices(), trial_vec.values(), trial_vec.curr_size(), trial_hashes);
#ifdef USE_MPI
        MPI_Gather(&matr_el, 1, MPI_DOUBLE, recv_nums, 1, MPI_DOUBLE, hf_proc, MPI_COMM_WORLD);
        MPI_Gather(&denom, 1, MPI_DOUBLE, recv_dens, 1, MPI_DOUBLE, hf_proc, MPI_COMM_WORLD);
#else
        recv_nums[0] = matr_el;
        recv_dens[0] = denom;
#endif
        if (proc_rank == hf_proc) {
            matr_el = 0;
            denom = 0;
            for (proc_idx = 0; proc_idx < n_procs; proc_idx++) {
                matr_el += recv_nums[proc_idx];
                denom += recv_dens[proc_idx];
            }
            fprintf(num_file, "%lf\n", matr_el);
            fprintf(den_file, "%lf\n", denom);
            printf("%6u, n walk: %7u, en est: %lf, shift: %lf\n", iterat, (unsigned int)glob_norm, matr_el / denom, en_shift);
            fprintf(ini_file, "%zu\n", n_ini);
        }
        
        // Save vector snapshot to disk
        if ((iterat + 1) % save_interval == 0) {
            sol_vec.save(result_dir);
            if (proc_rank == hf_proc) {
                fflush(num_file);
                fflush(den_file);
                fflush(shift_file);
                fflush(nonz_file);
                fflush(walk_file);
            }
        }
    }
    sol_vec.save(result_dir);
    if (proc_rank == hf_proc) {
        fclose(num_file);
        fclose(den_file);
        fclose(shift_file);
        fclose(nonz_file);
        fclose(walk_file);
    }
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}

