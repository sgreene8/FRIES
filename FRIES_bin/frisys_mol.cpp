/*! \file
 *
 * \brief FRI algorithm with systematic matrix compression for a molecular
 * system
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <FRIES/Hamiltonians/near_uniform.hpp>
#include <FRIES/io_utils.hpp>
#include <FRIES/Ext_Libs/dcmt/dc.h>
#include <FRIES/compress_utils.hpp>
#include <FRIES/Ext_Libs/argparse.h>
#include <FRIES/Hamiltonians/heat_bathPP.hpp>
#include <FRIES/Hamiltonians/molecule.hpp>
#define max_iter 1000000

static const char *const usage[] = {
    "frisys_mol [options] [[--] args]",
    "frisys_mol [options]",
    NULL,
};

int main(int argc, const char * argv[]) {
    const char *hf_path = NULL;
    const char *result_dir = "./";
    const char *load_dir = NULL;
    const char *ini_path = NULL;
    const char *trial_path = NULL;
    const char *determ_path = NULL;
    unsigned int target_nonz = 0;
    unsigned int matr_samp = 0;
    unsigned int max_n_dets = 0;
    unsigned int init_thresh = 0;
    unsigned int tmp_norm = 0;
    struct argparse_option options[] = {
        OPT_HELP(),
        OPT_STRING('d', "hf_path", &hf_path, "Path to the directory that contains the HF output files eris.txt, hcore.txt, symm.txt, hf_en.txt, and sys_params.txt"),
        OPT_INTEGER('t', "target", &tmp_norm, "Target one-norm of solution vector"),
        OPT_INTEGER('m', "vec_nonz", &target_nonz, "Target number of nonzero vector elements to keep after each iteration"),
        OPT_INTEGER('M', "mat_nonz", &matr_samp, "Target number of nonzero matrix elements to keep after each iteration"),
        OPT_STRING('y', "result_dir", &result_dir, "Directory in which to save output files"),
        OPT_INTEGER('p', "max_dets", &max_n_dets, "Maximum number of determinants on a single MPI process."),
        OPT_INTEGER('i', "initiator", &init_thresh, "Magnitude of vector element required to make the corresponding determinant an initiator."),
        OPT_STRING('l', "load_dir", &load_dir, "Directory from which to load checkpoint files from a previous systematic FRI calculation (in binary format, see documentation for DistVec::save() and DistVec::load())."),
        OPT_STRING('n', "ini_vec", &ini_path, "Prefix for files containing the vector with which to initialize the calculation (files must have names <ini_vec>dets and <ini_vec>vals and be text files)."),
        OPT_STRING('t', "trial_vec", &trial_path, "Prefix for files containing the vector with which to calculate the energy (files must have names <trial_vec>dets and <trial_vec>vals and be text files)."),
        OPT_STRING('s', "det_space", &determ_path, "Path to a .txt file containing the determinants used to define the deterministic space to use in a semistochastic calculation."),
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
    
    double target_norm = tmp_norm;
    
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
    unsigned int save_interval = 100;
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
    unsigned int spawn_length = matr_samp * 5 / n_procs;
    size_t adder_size = spawn_length > 1000000 ? 1000000 : spawn_length;
    DistVec<double> sol_vec(max_n_dets, adder_size, rngen_ptr, n_orb * 2, n_elec_unf, n_procs, 0);
    size_t det_size = CEILING(2 * n_orb, 8);
    size_t det_idx;
    
    Matrix<uint8_t> symm_lookup(n_irreps, n_orb + 1);
    gen_symm_lookup(symm, symm_lookup);
    unsigned int max_n_symm = 0;
    for (det_idx = 0; det_idx < n_irreps; det_idx++) {
        if (symm_lookup[det_idx][0] > max_n_symm) {
            max_n_symm = symm_lookup[det_idx][0];
        }
    }
    
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
        MPI_Bcast(proc_scrambler, 2 * n_orb, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
#endif
    }
    sol_vec.proc_scrambler_ = proc_scrambler;
    
    uint8_t hf_det[det_size];
    gen_hf_bitstring(n_orb, n_elec - n_frz, hf_det);
    hf_proc = sol_vec.idx_to_proc(hf_det);
    
    uint8_t tmp_orbs[n_elec_unf];
    uint8_t (*orb_indices1)[4] = (uint8_t (*)[4])malloc(sizeof(char) * 4 * spawn_length);
    
    size_t n_trial;
    size_t n_ex = n_orb * n_orb * n_elec_unf * n_elec_unf;
    DistVec<double> trial_vec(100, 100, rngen_ptr, n_orb * 2, n_elec_unf, n_procs, 0);
    DistVec<double> htrial_vec(100 * n_ex, 100 * n_ex, rngen_ptr, n_orb * 2, n_elec_unf, n_procs, 0);
    trial_vec.proc_scrambler_ = proc_scrambler;
    htrial_vec.proc_scrambler_ = proc_scrambler;
    if (trial_path) { // load trial vector from file
        Matrix<uint8_t> &load_dets = sol_vec.indices();
        double *load_vals = (double *)sol_vec.values();
        
        n_trial = load_vec_txt(trial_path, load_dets, load_vals, DOUB);
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
    unsigned long long *trial_hashes = (unsigned long long *)malloc(sizeof(long long) * trial_vec.curr_size());
    for (det_idx = 0; det_idx < trial_vec.curr_size(); det_idx++) {
        trial_hashes[det_idx] = sol_vec.idx_to_hash(trial_vec.indices()[det_idx]);
    }
    
    // Calculate H * trial vector, and accumulate results on each processor
    h_op(htrial_vec, symm, tot_orb, *eris, *h_core, (uint8_t *)orb_indices1, n_frz, n_elec_unf, 0, 1, hf_en);
    htrial_vec.collect_procs();
    unsigned long long *htrial_hashes = (unsigned long long *)malloc(sizeof(long long) * htrial_vec.curr_size());
    for (det_idx = 0; det_idx < htrial_vec.curr_size(); det_idx++) {
        htrial_hashes[det_idx] = sol_vec.idx_to_hash(htrial_vec.indices()[det_idx]);
    }
    
    // Count # single/double excitations from HF
    sol_vec.gen_orb_list(hf_det, tmp_orbs);
    size_t n_hf_doub = doub_ex_symm(hf_det, tmp_orbs, n_elec_unf, n_orb, orb_indices1, symm);
    size_t n_hf_sing = count_singex(hf_det, tmp_orbs, symm, n_orb, symm_lookup, n_elec_unf);
    double p_doub = (double) n_hf_doub / (n_hf_sing + n_hf_doub);
    
    char file_path[100];
    FILE *num_file = NULL;
    FILE *den_file = NULL;
    FILE *shift_file = NULL;
    FILE *norm_file = NULL;
    FILE *nkept_file = NULL;
    
    // Initialize solution vector
    if (load_dir) {
        sol_vec.load(load_dir);
        
        // load energy shift (see https://stackoverflow.com/questions/13790662/c-read-only-last-line-of-a-file-no-loops)
        static const long max_len = 20;
        sprintf(file_path, "%sS.txt", load_dir);
        shift_file = fopen(file_path, "rb");
        fseek(shift_file, -max_len, SEEK_END);
        fread(file_path, max_len, 1, shift_file);
        fclose(shift_file);
        shift_file = NULL;
        
        file_path[max_len - 1] = '\0';
        char *last_newline = strrchr(file_path, '\n');
        char *last_line = last_newline + 1;
        
        sscanf(last_line, "%lf", &en_shift);
    }
    else if (ini_path) {
        Matrix<uint8_t> &load_dets = sol_vec.indices();
        double *load_vals = (double *)sol_vec.values();
        
        size_t n_dets = load_vec_txt(ini_path, load_dets, load_vals, DOUB);
        
        for (det_idx = 0; det_idx < n_dets; det_idx++) {
            sol_vec.add(load_dets[det_idx], load_vals[det_idx], 1);
        }
    }
    else {
        if (hf_proc == proc_rank) {
            sol_vec.add(hf_det, 100, 1);
        }
    }
    sol_vec.perform_add();
    loc_norm = sol_vec.local_norm();
    sum_mpi(loc_norm, &glob_norm, proc_rank, n_procs);
    if (load_dir) {
        last_norm = glob_norm;
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
        strcat(file_path, "norm.txt");
        norm_file = fopen(file_path, "a");
        strcpy(file_path, result_dir);
        strcat(file_path, "nkept.txt");
        nkept_file = fopen(file_path, "a");
        
        strcpy(file_path, result_dir);
        strcat(file_path, "params.txt");
        FILE *param_f = fopen(file_path, "w");
        fprintf(param_f, "FRI calculation\nHF path: %s\nepsilon (imaginary time step): %lf\nTarget norm %lf\nInitiator threshold: %u\nMatrix nonzero: %u\nVector nonzero: %u\n", hf_path, eps, target_norm, init_thresh, matr_samp, target_nonz);
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
    
    Matrix<double> subwt_mem(spawn_length, n_orb);
    unsigned int *ndiv_vec = (unsigned int *)malloc(sizeof(unsigned int) * spawn_length);
    double *comp_vec1 = (double *)malloc(sizeof(double) * spawn_length);
    double *comp_vec2 = (double *)malloc(sizeof(double) * spawn_length);
    size_t (*comp_idx)[2] = (size_t (*)[2])malloc(sizeof(size_t) * 2 * spawn_length);
    size_t comp_len;
    size_t *det_indices1 = (size_t *)malloc(sizeof(size_t) * 2 * spawn_length);
    size_t *det_indices2 = &det_indices1[spawn_length];
    uint8_t (*orb_indices2)[4] = (uint8_t (*)[4])malloc(sizeof(uint8_t) * 4 * spawn_length);
    unsigned int unocc_symm_cts[n_irreps][2];
    Matrix<int> keep_idx(spawn_length, n_orb);
    double *wt_remain = (double *)calloc(spawn_length, sizeof(double));
    size_t samp_idx, weight_idx;
    
    hb_info *hb_probs = set_up(tot_orb, n_orb, *eris);
    
    double last_one_norm = 0;
    double matr_el;
    double recv_nums[n_procs];
    double recv_dens[n_procs];
    
    // Parameters for systematic sampling
    double rn_sys = 0;
    double weight;
    int glob_n_nonz; // Number of nonzero elements in whole vector (across all processors)
    double loc_norms[n_procs];
    size_t *srt_arr = (size_t *)malloc(sizeof(size_t) * max_n_dets);
    for (det_idx = 0; det_idx < max_n_dets; det_idx++) {
        srt_arr[det_idx] = det_idx;
    }
    int *keep_exact = (int *)calloc(max_n_dets, sizeof(int));
    size_t n_subwt;
    
    unsigned int iterat;
    for (iterat = 0; iterat < max_iter; iterat++) {
        sum_mpi(sol_vec.n_nonz(), &glob_n_nonz, proc_rank, n_procs);
        
        // Systematic matrix compression
        if (glob_n_nonz > matr_samp) {
            fprintf(stderr, "Warning: target number of matrix samples (%u) is less than number of nonzero vector elements (%d)\n", matr_samp, glob_n_nonz);
        }
        
        // Singles vs doubles
        n_subwt = 2;
        subwt_mem.reshape(spawn_length, 2);
        keep_idx.reshape(spawn_length, 2);
        for (det_idx = 0; det_idx < sol_vec.curr_size(); det_idx++) {
            double *curr_el = sol_vec[det_idx];
            weight = fabs(*curr_el);
            comp_vec1[det_idx] = weight;
            if (weight > 0) {
                subwt_mem(det_idx, 0) = p_doub;
                subwt_mem(det_idx, 1) = (1 - p_doub);
                ndiv_vec[det_idx] = 0;
            }
            else {
                ndiv_vec[det_idx] = 1;
            }
        }
        if (proc_rank == 0) {
            rn_sys = genrand_mt(rngen_ptr) / (1. + UINT32_MAX);
        }
        comp_len = comp_sub(comp_vec1, sol_vec.curr_size(), ndiv_vec, n_subwt, subwt_mem, keep_idx, matr_samp, wt_remain, rn_sys, comp_vec2, comp_idx);
        
        // First occupied orbital
        n_subwt = n_elec_unf;
        subwt_mem.reshape(spawn_length, n_elec_unf);
        keep_idx.reshape(spawn_length, n_elec_unf);
        for (samp_idx = 0; samp_idx < comp_len; samp_idx++) {
            det_idx = comp_idx[samp_idx][0];
            det_indices1[samp_idx] = det_idx;
            orb_indices1[samp_idx][0] = comp_idx[samp_idx][1];
            if (orb_indices1[samp_idx][0] == 0) { // double excitation
                ndiv_vec[samp_idx] = 0;
//                comp_vec2[samp_idx] *= calc_o1_probs(hb_probs, subwt_mem[samp_idx], n_elec_unf, orbs_at_pos(sol_vec, det_idx));
                calc_o1_probs(hb_probs, subwt_mem[samp_idx], n_elec_unf, sol_vec.orbs_at_pos(det_idx));
            }
            else {
                uint8_t *occ_orbs = sol_vec.orbs_at_pos(det_idx);
                count_symm_virt(unocc_symm_cts, occ_orbs, n_elec_unf, n_orb, n_irreps, symm_lookup, symm);
                unsigned int n_occ = count_sing_allowed(occ_orbs, n_elec_unf, symm, n_orb, unocc_symm_cts);
                if (n_occ == 0) {
                    ndiv_vec[samp_idx] = 1;
                    comp_vec2[samp_idx] = 0;
                }
                else {
                    ndiv_vec[samp_idx] = n_occ;
                }
            }
        }
        
        if (proc_rank == 0) {
            rn_sys = genrand_mt(rngen_ptr) / (1. + UINT32_MAX);
        }
        comp_len = comp_sub(comp_vec2, comp_len, ndiv_vec, n_subwt, subwt_mem, keep_idx, matr_samp, wt_remain, rn_sys, comp_vec1, comp_idx);
        
        // Unoccupied orbital (single); 2nd occupied (double)
        n_subwt = n_elec_unf;
        for (samp_idx = 0; samp_idx < comp_len; samp_idx++) {
            weight_idx = comp_idx[samp_idx][0];
            det_idx = det_indices1[weight_idx];
            det_indices2[samp_idx] = det_idx;
            orb_indices2[samp_idx][0] = orb_indices1[weight_idx][0]; // single or double
            orb_indices2[samp_idx][1] = comp_idx[samp_idx][1]; // first occupied orbital index (converted to orbital below)
            uint8_t *occ_orbs = sol_vec.orbs_at_pos(det_idx);
            if (orb_indices2[samp_idx][0] == 0) { // double excitation
                ndiv_vec[samp_idx] = 0;
//                comp_vec1[samp_idx] *= calc_o2_probs(hb_probs, subwt_mem[samp_idx], n_elec_unf, occ_orbs, &orb_indices2[samp_idx][1]);
                calc_o2_probs(hb_probs, subwt_mem[samp_idx], n_elec_unf, occ_orbs, &orb_indices2[samp_idx][1]);
            }
            else { // single excitation
                count_symm_virt(unocc_symm_cts, occ_orbs, n_elec_unf, n_orb, n_irreps, symm_lookup, symm);
                unsigned int n_virt = count_sing_virt(occ_orbs, n_elec_unf, symm, n_orb, unocc_symm_cts, &orb_indices2[samp_idx][1]);
                if (n_virt == 0) {
                    ndiv_vec[samp_idx] = 1;
                    comp_vec1[samp_idx] = 0;
                }
                else {
                    ndiv_vec[samp_idx] = n_virt;
                    orb_indices2[samp_idx][3] = n_virt; // number of allowed virtual orbitals
                }
            }
        }
        if (proc_rank == 0) {
            rn_sys = genrand_mt(rngen_ptr) / (1. + UINT32_MAX);
        }
        comp_len = comp_sub(comp_vec1, comp_len, ndiv_vec, n_subwt, subwt_mem, keep_idx, matr_samp, wt_remain, rn_sys, comp_vec2, comp_idx);
        
        // 1st unoccupied (double)
        n_subwt = n_orb;
        subwt_mem.reshape(spawn_length, n_orb);
        keep_idx.reshape(spawn_length, n_orb);
        for (samp_idx = 0; samp_idx < comp_len; samp_idx++) {
            weight_idx = comp_idx[samp_idx][0];
            det_idx = det_indices2[weight_idx];
            det_indices1[samp_idx] = det_idx;
            orb_indices1[samp_idx][0] = orb_indices2[weight_idx][0]; // single or double
            uint8_t o1_orb = orb_indices2[weight_idx][1];
            orb_indices1[samp_idx][1] = o1_orb; // 1st occupied orbital
            orb_indices1[samp_idx][2] = comp_idx[samp_idx][1]; // 2nd occupied orbital index (doubles), converted to orbital below; unoccupied orbital index (singles)
            if (orb_indices1[samp_idx][0] == 0) { // double excitation
                ndiv_vec[samp_idx] = 0;
                uint8_t *occ_tmp = sol_vec.orbs_at_pos(det_idx);
                orb_indices1[samp_idx][2] = occ_tmp[orb_indices1[samp_idx][2]];
//                comp_vec2[samp_idx] *= calc_u1_probs(hb_probs, subwt_mem[samp_idx], o1_orb, sol_vec->indices[det_indices1[samp_idx]]);
                calc_u1_probs(hb_probs, subwt_mem[samp_idx], o1_orb, sol_vec.indices()[det_idx]);
            }
            else { // single excitation
                orb_indices1[samp_idx][3] = orb_indices2[weight_idx][3];
                ndiv_vec[samp_idx] = 1;
            }
        }
        if (proc_rank == 0) {
            rn_sys = genrand_mt(rngen_ptr) / (1. + UINT32_MAX);
        }
        comp_len = comp_sub(comp_vec2, comp_len, ndiv_vec, n_subwt, subwt_mem, keep_idx, matr_samp, wt_remain, rn_sys, comp_vec1, comp_idx);
        
        // 2nd unoccupied (double)
        n_subwt = max_n_symm;
        subwt_mem.reshape(spawn_length, max_n_symm);
        keep_idx.reshape(spawn_length, max_n_symm);
        for (samp_idx = 0; samp_idx < comp_len; samp_idx++) {
            weight_idx = comp_idx[samp_idx][0];
            det_idx = det_indices1[weight_idx];
            det_indices2[samp_idx] = det_idx;
            orb_indices2[samp_idx][0] = orb_indices1[weight_idx][0]; // single or double
            uint8_t o1_orb = orb_indices1[weight_idx][1];
            orb_indices2[samp_idx][1] = o1_orb; // 1st occupied orbital
            uint8_t o2_orb = orb_indices1[weight_idx][2];
            orb_indices2[samp_idx][2] = o2_orb; // 2nd occupied orbital (doubles); unoccupied orbital index (singles)
            if (orb_indices2[samp_idx][0] == 0) { // double excitation
                uint8_t u1_orb = comp_idx[samp_idx][1] + n_orb * (o1_orb / n_orb);
                if (read_bit(sol_vec.indices()[det_idx], u1_orb)) {
                    comp_vec1[samp_idx] = 0;
                }
                else {
                    ndiv_vec[samp_idx] = 0;
                    orb_indices2[samp_idx][3] = u1_orb;
//                comp_vec1[samp_idx] *= calc_u2_probs(hb_probs, subwt_mem[samp_idx], o1_orb, o2_orb, u1_orb, (uint8_t *)symm_lookup, symm, max_n_symm); // not normalizing
                    double u2_norm = calc_u2_probs(hb_probs, subwt_mem[samp_idx], o1_orb, o2_orb, u1_orb, symm_lookup, symm, &max_n_symm);
                    if (u2_norm == 0) {
                        comp_vec1[samp_idx] = 0;
                    }
                }
            }
            else {
                orb_indices2[samp_idx][3] = orb_indices1[weight_idx][3];
                ndiv_vec[samp_idx] = 1;
            }
        }
        if (proc_rank == 0) {
            rn_sys = genrand_mt(rngen_ptr) / (1. + UINT32_MAX);
        }
        comp_len = comp_sub(comp_vec1, comp_len, ndiv_vec, n_subwt, subwt_mem, keep_idx, matr_samp, wt_remain, rn_sys, comp_vec2, comp_idx);
        
        uint8_t *spawn_dets = (uint8_t *)subwt_mem.data();
        size_t num_added = 0;
        keep_idx.reshape(spawn_length, 1);
        for (samp_idx = 0; samp_idx < comp_len; samp_idx++) {
            weight_idx = comp_idx[samp_idx][0];
            det_idx = det_indices2[weight_idx];
            uint8_t *curr_det = sol_vec.indices()[det_idx];
            double *curr_el = sol_vec[det_idx];
            int ini_flag = fabs(*curr_el) > init_thresh;
            int el_sign = 1 - 2 * (*curr_el < 0);
            uint8_t *occ_orbs = sol_vec.orbs_at_pos(det_idx);
            if (orb_indices2[weight_idx][0] == 0) { // double excitation
                uint8_t doub_orbs[4];
                doub_orbs[0] = orb_indices2[weight_idx][1];
                doub_orbs[1] = orb_indices2[weight_idx][2];
                doub_orbs[2] = orb_indices2[weight_idx][3];
                uint8_t u2_symm = symm[doub_orbs[0] % n_orb] ^ symm[doub_orbs[1] % n_orb] ^ symm[doub_orbs[2] % n_orb];
                doub_orbs[3] = symm_lookup[u2_symm][comp_idx[samp_idx][1] + 1] + n_orb * (doub_orbs[1] / n_orb);
                if (read_bit(curr_det, doub_orbs[3])) { // chosen orbital is occupied; unsuccessful spawn
                    continue;
                }
                if (doub_orbs[2] > doub_orbs[3]) {
                    uint8_t tmp = doub_orbs[3];
                    doub_orbs[3] = doub_orbs[2];
                    doub_orbs[2] = tmp;
                }
                if (doub_orbs[0] > doub_orbs[1]) {
                    uint8_t tmp = doub_orbs[1];
                    doub_orbs[1] = doub_orbs[0];
                    doub_orbs[0] = tmp;
                }
                matr_el = doub_matr_el_nosgn(doub_orbs, tot_orb, *eris, n_frz);
                if (fabs(matr_el) > 1e-9 && comp_vec2[samp_idx] > 1e-9) {
//                    matr_el *= -eps / p_doub / calc_unnorm_wt(hb_probs, doub_orbs) * el_sign * par_sign * comp_vec2[samp_idx];
                    matr_el *= -eps / p_doub / calc_norm_wt(hb_probs, doub_orbs, occ_orbs, n_elec_unf, curr_det, symm_lookup, symm) * el_sign * comp_vec2[samp_idx];
                    uint8_t *new_det = &spawn_dets[num_added * det_size];
                    memcpy(new_det, curr_det, det_size);
                    matr_el *= doub_det_parity(new_det, doub_orbs);
                    comp_vec1[num_added] = matr_el;
                    keep_idx(num_added, 0) = ini_flag;
                    num_added++;
                }
            }
            else { // single excitation
                uint8_t sing_orbs[2];
                uint8_t o1 = orb_indices2[weight_idx][1];
                sing_orbs[0] = o1;
                uint8_t u1_symm = symm[o1 % n_orb];
                sing_orbs[1] = virt_from_idx(curr_det, symm_lookup[u1_symm], n_orb * (o1 / n_orb), orb_indices2[weight_idx][2]);
                matr_el = sing_matr_el_nosgn(sing_orbs, occ_orbs, tot_orb, *eris, *h_core, n_frz, n_elec_unf);
                if (fabs(matr_el) > 1e-9 && comp_vec2[samp_idx] > 1e-9) {
                    count_symm_virt(unocc_symm_cts, occ_orbs, n_elec_unf, n_orb, n_irreps, symm_lookup, symm);
                    unsigned int n_occ = count_sing_allowed(occ_orbs, n_elec_unf, symm, n_orb, unocc_symm_cts);
                    uint8_t *new_det = &spawn_dets[num_added * det_size];
                    memcpy(new_det, curr_det, det_size);
                    matr_el *= -eps / (1 - p_doub) * n_occ * orb_indices2[weight_idx][3] * el_sign * sing_det_parity(new_det, sing_orbs) * comp_vec2[samp_idx];
                    comp_vec1[num_added] = matr_el;
                    keep_idx(num_added, 0) = ini_flag;
                    num_added++;
                }
            }
        }
        
        // Death/cloning step
        for (det_idx = 0; det_idx < sol_vec.curr_size(); det_idx++) {
            double *curr_el = sol_vec[det_idx];
            if (*curr_el != 0) {
                double *diag_el = sol_vec.matr_el_at_pos(det_idx);
                uint8_t *occ_orbs = sol_vec.orbs_at_pos(det_idx);
                if (isnan(*diag_el)) {
                    *diag_el = diag_matrel(occ_orbs, tot_orb, *eris, *h_core, n_frz, n_elec) - hf_en;
                }
                *curr_el *= 1 - eps * (*diag_el - en_shift);
            }
        }
        
        comp_len = num_added;
        num_added = 0;
        int glob_adding = 1;
        samp_idx = 0;
        while (glob_adding) {
            while (samp_idx < comp_len) {
                if (num_added >= adder_size) {
                    break;
                }
                int ini_flag = keep_idx(samp_idx, 0);
                keep_idx(samp_idx, 0) = 0;
                sol_vec.add(&spawn_dets[samp_idx * det_size], comp_vec1[samp_idx], ini_flag);
                num_added++;
                samp_idx++;
            }
            sol_vec.perform_add();
            loc_norms[proc_rank] = num_added;
#ifdef USE_MPI
            MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, loc_norms, 1, MPI_DOUBLE, MPI_COMM_WORLD);
#endif
            glob_adding = 0;
            for (proc_idx = 0; proc_idx < n_procs; proc_idx++) {
                if (loc_norms[proc_idx] > 0) {
                    glob_adding = 1;
                    break;
                }
            }
            num_added = 0;
        }
        
        // Compression step
        unsigned int n_samp = target_nonz;
        loc_norms[proc_rank] = find_preserve(sol_vec.values(), srt_arr, keep_exact, sol_vec.curr_size(), &n_samp, &glob_norm);
        if (proc_rank == hf_proc) {
            fprintf(nkept_file, "%u\n", target_nonz - n_samp);
        }
        
        // Adjust shift
        if ((iterat + 1) % shift_interval == 0) {
            adjust_shift(&en_shift, glob_norm, &last_one_norm, target_norm, shift_damping / shift_interval / eps);
            if (proc_rank == hf_proc) {
                fprintf(shift_file, "%lf\n", en_shift);
                fprintf(norm_file, "%lf\n", glob_norm);
            }
        }
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
            printf("%6u, en est: %.9lf, shift: %lf, norm: %lf\n", iterat, matr_el / denom, en_shift, glob_norm);
        }
        
        if (proc_rank == 0) {
            rn_sys = genrand_mt(rngen_ptr) / (1. + UINT32_MAX);
        }
#ifdef USE_MPI
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, loc_norms, 1, MPI_DOUBLE, MPI_COMM_WORLD);
#endif
        sys_comp(sol_vec.values(), sol_vec.curr_size(), loc_norms, n_samp, keep_exact, rn_sys);
        for (det_idx = 0; det_idx < sol_vec.curr_size(); det_idx++) {
            if (keep_exact[det_idx] && sol_vec.indices()[det_idx] != hf_det) {
                sol_vec.del_at_pos(det_idx);
                keep_exact[det_idx] = 0;
            }
        }
        
        if ((iterat + 1) % save_interval == 0) {
            sol_vec.save(result_dir);
            if (proc_rank == hf_proc) {
                fflush(num_file);
                fflush(den_file);
                fflush(shift_file);
                fflush(nkept_file);
            }
        }
    }
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}

