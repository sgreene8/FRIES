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
#include <cinttypes>
#include <FRIES/Hamiltonians/near_uniform.hpp>
#include <FRIES/io_utils.hpp>
#include <FRIES/Ext_Libs/dcmt/dc.h>
#include <FRIES/compress_utils.hpp>
#include <FRIES/Ext_Libs/argparse.hpp>
#include <FRIES/Hamiltonians/heat_bathPP.hpp>
#include <FRIES/Hamiltonians/molecule.hpp>
#include <stdexcept>

struct MyArgs : public argparse::Args {
    std::string hf_path = kwarg("hf_path", "Path to the directory that contains the HF output files eris.txt, hcore.txt, symm.txt, hf_en.txt, and sys_params.txt");
    double target_norm = kwarg("target", "Target one-norm of solution vector").set_default(0);
    std::string dist_str = kwarg("distribution", "Hamiltonian factorization to use, either heat-bath Power-Pitzer (HB) or unnormalized heat-bath Power-Pitzer (HB_unnorm)");
    uint32_t max_iter = kwarg("max_iter", "Maximum number of iterations to run the calculation").set_default(1000000);
    uint32_t target_nonz = kwarg("vec_nonz", "Target number of nonzero vector elements to keep after each iteration");
    uint32_t matr_samp = kwarg("mat_nonz", "Target number of nonzero matrix elements to keep after each multiplication by a Hamiltonian matrix factor");
    std::string result_dir = kwarg("result_dir", "Directory in which to save output files").set_default<std::string>("./");
    uint32_t max_n_dets = kwarg("max_dets", "Maximum number of determinants on a single MPI process");
    double init_thresh = kwarg("initiator", "Magnitude of vector element required to make it an initiator").set_default(0);
    std::shared_ptr<std::string> load_dir = kwarg("load_dir", "Directory from which to load checkpoint files from a previous FRI calculation (in binary format, see documentation for DistVec::save() and DistVec::load())");
    std::shared_ptr<std::string> ini_path = kwarg("ini_vec", "Prefix for files containing the vector with which to initialize the calculation (files must have names <ini_vec>dets and <ini_vec>vals and be text files)");
    std::shared_ptr<std::string> trial_path = kwarg("trial_vec", "Prefix for files containing the vector with which to calculate the energy (files must have names <trial_vec>dets and <trial_vec>vals and be text)");
    std::shared_ptr<std::string> determ_path = kwarg("det_space", "Path to a .txt file containing the determinants used to define the deterministic space to use in a semistochastic calculation.");
    double pt_weight = kwarg("unbias", "The prefactor for adding corrections to the initiator bias.").set_default(0);
    
    CONSTRUCTOR(MyArgs);
};

int main(int argc, char * argv[]) {
    MyArgs args(argc, argv);
    
    h_dist qmc_dist;
    try {
        if (args.dist_str == "HB") {
            qmc_dist = heat_bath;
        }
        else if (args.dist_str == "HB_unnorm") {
            qmc_dist = unnorm_heat_bath;
        }
        else {
            throw std::runtime_error("\"dist_str\" argument must be either \"NU\" or \"HB_unnorm\"");
        }
    } catch (std::exception &ex) {
        std::cerr << "\nError parsing command line: " << ex.what() << "\n\n";
        return 1;
    }
    int new_hb = qmc_dist == unnorm_heat_bath;
    
    double target_norm = args.target_norm;
    
    try {
        int n_procs = 1;
        int proc_rank = 0;
        unsigned int hf_proc;
#ifdef USE_MPI
        MPI_Init(NULL, NULL);
        MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
#endif
        
        uint32_t max_n_dets = args.max_n_dets;
        const char *result_dir = args.result_dir.c_str();
        uint32_t matr_samp = args.matr_samp;
        
        // Parameters
        double shift_damping = 0.05;
        unsigned int shift_interval = 10;
        unsigned int save_interval = 100;
        double en_shift = 0;
        
        // Read in data files
        hf_input in_data;
        parse_hf_input(args.hf_path.c_str(), &in_data);
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
        unsigned int spawn_length = args.matr_samp * 4 / n_procs;
        size_t adder_size = spawn_length > 1000000 ? 1000000 : spawn_length;
        std::function<double(const uint8_t *)> diag_shortcut = [tot_orb, eris, h_core, n_frz, n_elec, hf_en](const uint8_t *occ_orbs) {
            return diag_matrel(occ_orbs, tot_orb, *eris, *h_core, n_frz, n_elec) - hf_en;
        };
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
        std::vector<uint32_t> proc_scrambler(2 * n_orb);
        double loc_norm, glob_norm;
        double last_norm = 0;
        
        if (args.load_dir != nullptr) {
            load_proc_hash(args.load_dir->c_str(), proc_scrambler.data());
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
        
        DistVec<double> sol_vec(max_n_dets, adder_size, n_orb * 2, n_elec_unf, n_procs, diag_shortcut, &en_shift, 2, proc_scrambler, vec_scrambler);
        
        uint8_t hf_det[det_size];
        gen_hf_bitstring(n_orb, n_elec - n_frz, hf_det);
        hf_proc = sol_vec.idx_to_proc(hf_det);
        
        uint8_t tmp_orbs[n_elec_unf];
        uint8_t (*orb_indices1)[4] = (uint8_t (*)[4])malloc(sizeof(char) * 4 * spawn_length);
        
# pragma mark Set up trial vector
        size_t n_trial;
        size_t n_ex = n_orb * n_orb * n_elec_unf * n_elec_unf;
        Matrix<uint8_t> &load_dets = sol_vec.indices();
        double *load_vals = (double *)sol_vec.values();
        if (args.trial_path != nullptr) { // load trial vector from file
            n_trial = load_vec_txt(args.trial_path->c_str(), load_dets, load_vals, DOUB);
        }
        else {
            n_trial = 1;
        }
        unsigned int tot_trial = sum_mpi((int) n_trial, proc_rank, n_procs);
        tot_trial = CEILING(tot_trial * 2, n_procs);
        DistVec<double> trial_vec(tot_trial, tot_trial, n_orb * 2, n_elec_unf, n_procs, proc_scrambler, vec_scrambler);
        DistVec<double> htrial_vec(tot_trial * n_ex / n_procs, tot_trial * n_ex / n_procs, n_orb * 2, n_elec_unf, n_procs, diag_shortcut, NULL, 2, proc_scrambler, vec_scrambler);
        if (args.trial_path != nullptr) { // load trial vector from file
            for (det_idx = 0; det_idx < n_trial; det_idx++) {
                trial_vec.add(load_dets[det_idx], load_vals[det_idx], 1);
                htrial_vec.add(load_dets[det_idx], load_vals[det_idx], 1);
            }
        }
        else { // Otherwise, use HF as trial vector
            if (hf_proc == proc_rank) {
                trial_vec.add(hf_det, 1, 1);
                htrial_vec.add(hf_det, 1, 1);
            }
        }
        trial_vec.perform_add();
        htrial_vec.perform_add();
        
        trial_vec.collect_procs();
        uintmax_t *trial_hashes = (uintmax_t *)malloc(sizeof(uintmax_t) * trial_vec.curr_size());
        for (det_idx = 0; det_idx < trial_vec.curr_size(); det_idx++) {
            trial_hashes[det_idx] = sol_vec.idx_to_hash(trial_vec.indices()[det_idx], tmp_orbs);
        }
        
        // Calculate H * trial vector, and accumulate results on each processor
        h_op_offdiag(htrial_vec, symm, tot_orb, *eris, *h_core, (uint8_t *)orb_indices1, n_frz, n_elec_unf, 1, 1);
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
        size_t n_hf_doub = doub_ex_symm(hf_det, tmp_orbs, n_elec_unf, n_orb, orb_indices1, symm);
        size_t n_hf_sing = count_singex(hf_det, tmp_orbs, symm, n_orb, symm_lookup, n_elec_unf);
        double p_doub = (double) n_hf_doub / (n_hf_sing + n_hf_doub);
        
        char file_path[300];
        FILE *num_file = NULL;
        FILE *den_file = NULL;
        FILE *shift_file = NULL;
        FILE *norm_file = NULL;
        FILE *nkept_file = NULL;
        FILE *ini_file = NULL;
        
        size_t n_determ = 0; // Number of deterministic determinants on this process
        if (args.load_dir == nullptr && args.determ_path != nullptr) {
            n_determ = sol_vec.init_dense(args.determ_path->c_str(), result_dir);
        }
        int dense_sizes[n_procs];
        int determ_tmp = (int) n_determ;
#ifdef USE_MPI
        MPI_Gather(&determ_tmp, 1, MPI_INT, dense_sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);
#else
        dense_sizes[proc_rank] = determ_tmp;
#endif
        if (proc_rank == 0 && args.load_dir == nullptr) {
            sprintf(file_path, "%sdense.txt", result_dir);
            FILE *dense_f = fopen(file_path, "w");
            if (!dense_f) {
                fprintf(stderr, "Error opening file containing sizes of deterministic subspaces.\n");
                return 0;
            }
            for (int proc_idx = 0; proc_idx < n_procs; proc_idx++) {
                fprintf(dense_f, "%d, ", dense_sizes[proc_idx]);
            }
            fprintf(dense_f, "\n");
            fclose(dense_f);
        }
        
#pragma mark Initialize solution vector
        if (args.load_dir != nullptr) {
            n_determ = sol_vec.load(args.load_dir->c_str());
            
            // load energy shift (see https://stackoverflow.com/questions/13790662/c-read-only-last-line-of-a-file-no-loops)
            static const long max_len = 20;
            sprintf(file_path, "%sS.txt", args.load_dir->c_str());
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
        else if (args.ini_path != nullptr) {
            Matrix<uint8_t> load_dets(max_n_dets, det_size);
            double *load_vals = sol_vec.values();
            
            size_t n_dets = load_vec_txt(args.ini_path->c_str(), load_dets, load_vals, DOUB);
            
            for (det_idx = 0; det_idx < n_dets; det_idx++) {
                sol_vec.add(load_dets[det_idx], load_vals[det_idx], 1);
            }
            n_dets++; // just to be safe
            bzero(load_vals, n_dets * sizeof(double));
        }
        else {
            if (hf_proc == proc_rank) {
                sol_vec.add(hf_det, 100, 1);
            }
        }
        sol_vec.perform_add();
        loc_norm = sol_vec.local_norm();
        glob_norm = sum_mpi(loc_norm, proc_rank, n_procs);
        if (args.load_dir != nullptr) {
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
            strcat(file_path, "nini.txt");
            ini_file = fopen(file_path, "a");
            
            strcpy(file_path, result_dir);
            strcat(file_path, "params.txt");
            FILE *param_f = fopen(file_path, "w");
            fprintf(param_f, "FRI calculation\nHF path: %s\nepsilon (imaginary time step): %lf\nTarget norm %lf\nInitiator threshold: %f\nMatrix nonzero: %u\nVector nonzero: %u\n", args.hf_path.c_str(), eps, target_norm, args.init_thresh, args.matr_samp, args.target_nonz);
            if (args.load_dir != nullptr) {
                fprintf(param_f, "Restarting calculation from %s\n", args.load_dir->c_str());
            }
            else if (args.ini_path != nullptr) {
                fprintf(param_f, "Initializing calculation from vector files with prefix %s\n", args.ini_path->c_str());
            }
            else {
                fprintf(param_f, "Initializing calculation from HF unit vector\n");
            }
            fclose(param_f);
        }
        
        size_t n_states = n_elec_unf > (n_orb - n_elec_unf / 2) ? n_elec_unf : n_orb - n_elec_unf / 2;
        Matrix<double> subwt_mem(spawn_length, n_states);
        uint16_t *sub_sizes = (uint16_t *)malloc(sizeof(uint16_t) * spawn_length);
        unsigned int *ndiv_vec = (unsigned int *)malloc(sizeof(unsigned int) * spawn_length);
        double *comp_vec1 = (double *)malloc(sizeof(double) * spawn_length);
        double *comp_vec2 = (double *)malloc(sizeof(double) * spawn_length);
        size_t (*comp_idx)[2] = (size_t (*)[2])malloc(sizeof(size_t) * 2 * spawn_length);
        size_t comp_len;
        size_t *det_indices1 = (size_t *)malloc(sizeof(size_t) * 2 * spawn_length);
        size_t *det_indices2 = &det_indices1[spawn_length];
        uint8_t (*orb_indices2)[4] = (uint8_t (*)[4])malloc(sizeof(uint8_t) * 4 * spawn_length);
        unsigned int unocc_symm_cts[n_irreps][2];
        Matrix<bool> keep_idx(spawn_length, n_states);
        double *wt_remain = (double *)calloc(spawn_length, sizeof(double));
        size_t samp_idx, weight_idx;
        
        hb_info *hb_probs = set_up(tot_orb, n_orb, *eris);
        
        double last_one_norm = 0;
        
        // Parameters for systematic sampling
        double rn_sys = 0;
        double weight;
        int glob_n_nonz; // Number of nonzero elements in whole vector (across all processors)
        double loc_norms[n_procs];
        max_n_dets = (unsigned int)sol_vec.max_size();
        size_t *srt_arr = (size_t *)malloc(sizeof(size_t) * max_n_dets);
        for (det_idx = 0; det_idx < max_n_dets; det_idx++) {
            srt_arr[det_idx] = det_idx;
        }
        std::vector<bool> keep_exact(max_n_dets, false);
        
#pragma mark Pre-calculate deterministic subspace of Hamiltonian
        size_t determ_h_size = n_determ * n_elec_unf * n_elec_unf * (n_orb - n_elec_unf / 2) * (n_orb - n_elec_unf / 2);
        unsigned int n_determ_h = 0;
        size_t *determ_from = (size_t *)malloc(determ_h_size * sizeof(size_t));
        Matrix<uint8_t> determ_to(determ_h_size, det_size);
        double *determ_matr_el = (double *)malloc(determ_h_size * sizeof(double));
        for (det_idx = 0; det_idx < n_determ; det_idx++) {
            uint8_t *curr_det = sol_vec.indices()[det_idx];
            uint8_t *occ_orbs = sol_vec.orbs_at_pos(det_idx);
            uint8_t (*sing_ex_orbs)[2] = (uint8_t (*)[2])orb_indices1;
            size_t ex_idx;
            
            size_t n_sing = sing_ex_symm(curr_det, occ_orbs, n_elec_unf, n_orb, sing_ex_orbs, symm);
            if (n_sing + n_determ_h > determ_h_size) {
                printf("Allocating more memory for deterministic part of Hamiltonian\n");
                determ_h_size *= 2;
                determ_from = (size_t *)realloc(determ_from, determ_h_size * sizeof(size_t));
                determ_to.reshape(determ_h_size, det_size);
                determ_matr_el = (double *)realloc(determ_matr_el, determ_h_size * sizeof(double));
            }
            for (ex_idx = 0; ex_idx < n_sing; ex_idx++) {
                double matr_el = sing_matr_el_nosgn(sing_ex_orbs[ex_idx], occ_orbs, tot_orb, *eris, *h_core, n_frz, n_elec_unf);
                uint8_t *new_det = determ_to[n_determ_h];
                memcpy(new_det, curr_det, det_size);
                matr_el *= sing_det_parity(new_det, sing_ex_orbs[ex_idx]) * -eps;
                
                determ_from[n_determ_h] = det_idx;
                determ_matr_el[n_determ_h] = matr_el;
                n_determ_h++;
            }
            
            uint8_t (*doub_ex_orbs)[4] = (uint8_t (*)[4])orb_indices1;
            size_t n_doub = doub_ex_symm(curr_det, occ_orbs, n_elec_unf, n_orb, doub_ex_orbs, symm);
            if (n_doub + n_determ_h > determ_h_size) {
                printf("Allocating more memory for deterministic part of Hamiltonian\n");
                determ_h_size *= 2;
                determ_from = (size_t *)realloc(determ_from, determ_h_size * sizeof(size_t));
                determ_to.reshape(determ_h_size, det_size);
                determ_matr_el = (double *)realloc(determ_matr_el, determ_h_size * sizeof(double));
            }
            for (ex_idx = 0; ex_idx < n_doub; ex_idx++) {
                double matr_el = doub_matr_el_nosgn(doub_ex_orbs[ex_idx], tot_orb, *eris, n_frz);
                uint8_t *new_det = determ_to[n_determ_h];
                memcpy(new_det, curr_det, det_size);
                matr_el *= doub_det_parity(new_det, doub_ex_orbs[ex_idx]) * -eps;
                
                determ_from[n_determ_h] = det_idx;
                determ_matr_el[n_determ_h] = matr_el;
                n_determ_h++;
            }
        }
        unsigned int tot_dense_h = sum_mpi((int) n_determ_h, proc_rank, n_procs);
        if (proc_rank == 0) {
            printf("Elements in dense H: %u\n", tot_dense_h);
        }
        
        unsigned int iterat;
        size_t n_ini;
        for (iterat = 0; iterat < args.max_iter; iterat++) {
            n_ini = 0;
            glob_n_nonz = sum_mpi(sol_vec.n_nonz(), proc_rank, n_procs);
            
            // Systematic matrix compression
            if (glob_n_nonz > args.matr_samp) {
                fprintf(stderr, "Warning: target number of matrix samples (%u) is less than number of nonzero vector elements (%d)\n", args.matr_samp, glob_n_nonz);
            }
            
#pragma mark Singles vs doubles
            subwt_mem.reshape(spawn_length, 2);
            keep_idx.reshape(spawn_length, 2);
            for (det_idx = n_determ; det_idx < sol_vec.curr_size(); det_idx++) {
                double *curr_el = sol_vec[det_idx];
                weight = fabs(*curr_el);
                n_ini += weight >= args.init_thresh;
                comp_vec1[det_idx - n_determ] = weight;
                if (weight > 0) {
                    subwt_mem(det_idx - n_determ, 0) = p_doub;
                    subwt_mem(det_idx - n_determ, 1) = (1 - p_doub);
                    ndiv_vec[det_idx - n_determ] = 0;
                }
                else {
                    ndiv_vec[det_idx - n_determ] = 1;
                }
            }
            if (proc_rank == 0) {
                rn_sys = genrand_mt(rngen_ptr) / (1. + UINT32_MAX);
            }
            comp_len = comp_sub(comp_vec1, sol_vec.curr_size() - n_determ, ndiv_vec, subwt_mem, keep_idx, NULL, matr_samp - tot_dense_h, wt_remain, rn_sys, comp_vec2, comp_idx);
            if (comp_len > spawn_length) {
                fprintf(stderr, "Error: insufficient memory allocated for matrix compression.\n");
            }
            
#pragma mark  First occupied orbital
            subwt_mem.reshape(spawn_length, n_elec_unf - new_hb);
            keep_idx.reshape(spawn_length, n_elec_unf - new_hb);
            for (samp_idx = 0; samp_idx < comp_len; samp_idx++) {
                det_idx = comp_idx[samp_idx][0] + n_determ;
                det_indices1[samp_idx] = det_idx;
                orb_indices1[samp_idx][0] = comp_idx[samp_idx][1];
                uint8_t *occ_orbs = sol_vec.orbs_at_pos(det_idx);
                if (orb_indices1[samp_idx][0] == 0) { // double excitation
                    ndiv_vec[samp_idx] = 0;
                    double tot_weight = calc_o1_probs(hb_probs, subwt_mem[samp_idx], n_elec_unf, occ_orbs, new_hb);
                    if (new_hb) {
                        comp_vec2[samp_idx] *= tot_weight;
                    }
                }
                else {
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
            comp_len = comp_sub(comp_vec2, comp_len, ndiv_vec, subwt_mem, keep_idx, NULL, matr_samp - tot_dense_h, wt_remain, rn_sys, comp_vec1, comp_idx);
            if (comp_len > spawn_length) {
                fprintf(stderr, "Error: insufficient memory allocated for matrix compression.\n");
            }
            
#pragma mark Unoccupied orbital (single); 2nd occupied (double)
            for (samp_idx = 0; samp_idx < comp_len; samp_idx++) {
                weight_idx = comp_idx[samp_idx][0];
                det_idx = det_indices1[weight_idx];
                det_indices2[samp_idx] = det_idx;
                orb_indices2[samp_idx][0] = orb_indices1[weight_idx][0]; // single or double
                orb_indices2[samp_idx][1] = comp_idx[samp_idx][1]; // first occupied orbital index (NOT converted to orbital below)
                if (orb_indices2[samp_idx][1] >= n_elec_unf) {
                    fprintf(stderr, "Error: chosen occupied orbital (first) is out of bounds\n");
                    comp_vec1[samp_idx] = 0;
                    ndiv_vec[samp_idx] = 1;
                    continue;
                }
                uint8_t *occ_orbs = sol_vec.orbs_at_pos(det_idx);
                if (orb_indices2[samp_idx][0] == 0) { // double excitation
                    ndiv_vec[samp_idx] = 0;
                    if (qmc_dist == heat_bath) {
                        calc_o2_probs(hb_probs, subwt_mem[samp_idx], n_elec_unf, occ_orbs, orb_indices2[samp_idx][1]);
                    }
                    else {
                        orb_indices2[samp_idx][1]++;
                        sub_sizes[samp_idx] = orb_indices2[samp_idx][1];
                        comp_vec1[samp_idx] *= calc_o2_probs_half(hb_probs, subwt_mem[samp_idx], n_elec_unf, occ_orbs, orb_indices2[samp_idx][1]);
                    }
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
            comp_len = comp_sub(comp_vec1, comp_len, ndiv_vec, subwt_mem, keep_idx, new_hb ? sub_sizes : NULL, matr_samp - tot_dense_h, wt_remain, rn_sys, comp_vec2, comp_idx);
            if (comp_len > spawn_length) {
                fprintf(stderr, "Error: insufficient memory allocated for matrix compression.\n");
            }
            
#pragma mark 1st unoccupied (double)
            subwt_mem.reshape(spawn_length, n_orb - n_elec_unf / 2);
            keep_idx.reshape(spawn_length, n_orb - n_elec_unf / 2);
            for (samp_idx = 0; samp_idx < comp_len; samp_idx++) {
                weight_idx = comp_idx[samp_idx][0];
                det_idx = det_indices2[weight_idx];
                det_indices1[samp_idx] = det_idx;
                orb_indices1[samp_idx][0] = orb_indices2[weight_idx][0]; // single or double
                uint8_t o1_idx = orb_indices2[weight_idx][1];
                orb_indices1[samp_idx][1] = o1_idx; // 1st occupied index
                uint8_t o2u1_orb = comp_idx[samp_idx][1]; // 2nd occupied orbital index (doubles), NOT converted to orbital below; unoccupied orbital index (singles)
                orb_indices1[samp_idx][2] = o2u1_orb;
                if (orb_indices1[samp_idx][0] == 0) { // double excitation
                    if (o2u1_orb >= n_elec_unf) {
                        fprintf(stderr, "Error: chosen occupied orbital (second) is out of bounds\n");
                        comp_vec2[samp_idx] = 0;
                        ndiv_vec[samp_idx] = 1;
                        continue;
                    }
                    ndiv_vec[samp_idx] = 0;
                    uint8_t *occ_tmp = sol_vec.orbs_at_pos(det_idx);
                    //                orb_indices1[samp_idx][2] = occ_tmp[o2u1_orb];
                    int o1_spin = o1_idx / (n_elec_unf / 2);
                    int o2_spin = occ_tmp[o2u1_orb] / n_orb;
                    double o1_orb = occ_tmp[o1_idx];
                    double tot_weight = calc_u1_probs(hb_probs, subwt_mem[samp_idx], o1_orb, occ_tmp, n_elec_unf, new_hb && (o1_spin == o2_spin));
                    if (new_hb) {
                        comp_vec2[samp_idx] *= tot_weight;
                    }
                }
                else { // single excitation
                    uint8_t n_virt = orb_indices2[weight_idx][3];
                    if (o2u1_orb >= n_virt) {
                        comp_vec1[samp_idx] = 0;
                        fprintf(stderr, "Error: index of chosen virtual orbital exceeds maximum\n");
                    }
                    orb_indices1[samp_idx][3] = n_virt;
                    ndiv_vec[samp_idx] = 1;
                }
            }
            if (proc_rank == 0) {
                rn_sys = genrand_mt(rngen_ptr) / (1. + UINT32_MAX);
            }
            comp_len = comp_sub(comp_vec2, comp_len, ndiv_vec, subwt_mem, keep_idx, NULL, matr_samp - tot_dense_h, wt_remain, rn_sys, comp_vec1, comp_idx);
            if (comp_len > spawn_length) {
                fprintf(stderr, "Error: insufficient memory allocated for matrix compression.\n");
            }
            
#pragma mark 2nd unoccupied (double)
            subwt_mem.reshape(spawn_length, max_n_symm);
            keep_idx.reshape(spawn_length, max_n_symm);
            for (samp_idx = 0; samp_idx < comp_len; samp_idx++) {
                weight_idx = comp_idx[samp_idx][0];
                det_idx = det_indices1[weight_idx];
                det_indices2[samp_idx] = det_idx;
                orb_indices2[samp_idx][0] = orb_indices1[weight_idx][0]; // single or double
                uint8_t o1_idx = orb_indices1[weight_idx][1];
                orb_indices2[samp_idx][1] = o1_idx; // 1st occupied index
                uint8_t o2_idx = orb_indices1[weight_idx][2];
                orb_indices2[samp_idx][2] = o2_idx; // 2nd occupied index (doubles); unoccupied orbital index (singles)
                if (orb_indices2[samp_idx][0] == 0) { // double excitation
                    uint8_t u1_orb = find_nth_virt(sol_vec.orbs_at_pos(det_idx), o1_idx / (n_elec_unf / 2), n_elec_unf, n_orb, comp_idx[samp_idx][1]);
                    uint8_t *curr_det = sol_vec.indices()[det_idx];
                    if (read_bit(curr_det, u1_orb)) { // now this really should never happen
                        fprintf(stderr, "Error: occupied orbital chosen as 1st virtual\n");
                        comp_vec1[samp_idx] = 0;
                        ndiv_vec[samp_idx] = 1;
                    }
                    else {
                        ndiv_vec[samp_idx] = 0;
                        orb_indices2[samp_idx][3] = u1_orb;
                        double tot_weight;
                        uint8_t *occ_tmp = sol_vec.orbs_at_pos(det_idx);
                        double o1_orb = occ_tmp[o1_idx];
                        double o2_orb = occ_tmp[o2_idx];
                        if (qmc_dist == heat_bath) {
                            tot_weight = calc_u2_probs(hb_probs, subwt_mem[samp_idx], o1_orb, o2_orb, u1_orb, symm_lookup, symm, &sub_sizes[samp_idx]);
                        }
                        else {
                            tot_weight = calc_u2_probs_half(hb_probs, subwt_mem[samp_idx], o1_orb, o2_orb, u1_orb, curr_det, symm_lookup, symm, &sub_sizes[samp_idx]);
                        }
                        if (new_hb || tot_weight == 0) {
                            comp_vec1[samp_idx] *= tot_weight;
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
            comp_len = comp_sub(comp_vec1, comp_len, ndiv_vec, subwt_mem, keep_idx, sub_sizes, matr_samp - tot_dense_h, wt_remain, rn_sys, comp_vec2, comp_idx);
            if (comp_len > spawn_length) {
                fprintf(stderr, "Error: insufficient memory allocated for matrix compression.\n");
            }
            
            double *vals_before_mult = sol_vec.values();
            sol_vec.set_curr_vec_idx(1);
            sol_vec.zero_vec();
            size_t vec_size = sol_vec.curr_size();
            if (args.pt_weight > 0) {
                sol_vec.zero_ini();
            }
            
            // The first time around, add only elements that came from noninitiators
            for (int add_ini = 0; add_ini < 2; add_ini++) {
                int num_added = 1;
                samp_idx = 0;
                while (num_added > 0) {
                    num_added = 0;
                    size_t start_idx = samp_idx;
                    while (samp_idx < comp_len && num_added < adder_size) {
                        weight_idx = comp_idx[samp_idx][0];
                        det_idx = det_indices2[weight_idx];
                        double curr_val = vals_before_mult[det_idx];
                        uint8_t ini_flag = fabs(curr_val) >= args.init_thresh;
                        if (ini_flag != add_ini) {
                            samp_idx++;
                            continue;
                        }
                        uint8_t *curr_det = sol_vec.indices()[det_idx];
                        uint8_t new_det[det_size];
                        //            int determ_flag = det_idx < n_determ;
                        uint8_t *occ_orbs = sol_vec.orbs_at_pos(det_idx);
                        uint8_t new_occ[n_elec_unf];
                        uint8_t o1_idx = orb_indices2[weight_idx][1];
                        orb_indices1[samp_idx][0] = 0; // 1 if this element is added, 0 otherwise
                        if (orb_indices2[weight_idx][0] == 0) { // double excitation
                            uint8_t o2_idx = orb_indices2[weight_idx][2];
                            uint8_t doub_orbs[4];
                            doub_orbs[0] = occ_orbs[o1_idx];
                            doub_orbs[1] = occ_orbs[o2_idx];
                            doub_orbs[2] = orb_indices2[weight_idx][3];
                            uint8_t u2_symm = symm[doub_orbs[0] % n_orb] ^ symm[doub_orbs[1] % n_orb] ^ symm[doub_orbs[2] % n_orb];
                            doub_orbs[3] = symm_lookup[u2_symm][comp_idx[samp_idx][1] + 1] + n_orb * (doub_orbs[1] / n_orb);
                            if (read_bit(curr_det, doub_orbs[3])) { // chosen orbital is occupied; unsuccessful spawn
                                if (new_hb) {
                                    fprintf(stderr, "Error: occupied orbital chosen as second virtual in unnormalized heat-bath\n");
                                }
                                comp_vec2[samp_idx] = 0;
                                samp_idx++;
                                continue;
                            }
                            if (doub_orbs[2] == doub_orbs[3]) { // This shouldn't happen, but in case it does (e.g. by numerical error)
                                fprintf(stderr, "Error: repeat virtual orbital chosen\n");
                                comp_vec2[samp_idx] = 0;
                                samp_idx++;
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
                                tmp = o1_idx;
                                o1_idx = o2_idx;
                                o2_idx = tmp;
                            }
                            double unsigned_mat = doub_matr_el_nosgn(doub_orbs, tot_orb, *eris, n_frz);
                            double tot_weight;
                            if (new_hb) {
                                tot_weight = calc_unnorm_wt(hb_probs, doub_orbs);
                            }
                            else {
                                tot_weight = calc_norm_wt(hb_probs, doub_orbs, occ_orbs, n_elec_unf, curr_det, symm_lookup, symm);
                            }
                            double add_el = unsigned_mat * -eps / p_doub / tot_weight * comp_vec2[samp_idx];
                            if (fabs(add_el) > 1e-9) {
                                if (curr_val < 0) {
                                    add_el *= -1;
                                }
                                memcpy(new_det, curr_det, det_size);
                                add_el *= doub_det_parity(new_det, doub_orbs);
                                doub_orbs[0] = o1_idx;
                                doub_orbs[1] = o2_idx;
                                doub_ex_orbs(occ_orbs, new_occ, doub_orbs, n_elec_unf);
                                det_indices1[samp_idx] = sol_vec.add(new_det, new_occ, add_el, ini_flag, (int *)&orb_indices1[samp_idx][1]);
                                orb_indices1[samp_idx][0] = !ini_flag;
                                num_added++;
                            }
                        }
                        else { // single excitation
                            uint8_t sing_orbs[2];
                            uint8_t o1 = occ_orbs[o1_idx];
                            sing_orbs[0] = o1;
                            uint8_t u1_symm = symm[o1 % n_orb];
                            uint8_t spin = o1 / n_orb;
                            sing_orbs[1] = virt_from_idx(curr_det, symm_lookup[u1_symm], n_orb * spin, orb_indices2[weight_idx][2]);
                            double unsigned_mat = sing_matr_el_nosgn(sing_orbs, occ_orbs, tot_orb, *eris, *h_core, n_frz, n_elec_unf);
                            if (fabs(unsigned_mat) > 1e-9) {
                                count_symm_virt(unocc_symm_cts, occ_orbs, n_elec_unf, n_orb, n_irreps, symm_lookup, symm);
                                unsigned int n_occ = count_sing_allowed(occ_orbs, n_elec_unf, symm, n_orb, unocc_symm_cts);
                                memcpy(new_det, curr_det, det_size);
                                double add_el = unsigned_mat * -eps / (1 - p_doub) * n_occ * orb_indices2[weight_idx][3] * sing_det_parity(new_det, sing_orbs) * comp_vec2[samp_idx];
                                if (sing_orbs[1] == 255) {
                                    fprintf(stderr, "Error: virtual orbital not found\n");
                                    comp_vec2[samp_idx] = 0;
                                    samp_idx++;
                                    continue;
                                }
                                if (curr_val < 0) {
                                    add_el *= -1;
                                }
                                sing_orbs[0] = o1_idx;
                                sing_ex_orbs(occ_orbs, new_occ, sing_orbs, n_elec_unf);
                                det_indices1[samp_idx] = sol_vec.add(new_det, new_occ, add_el, ini_flag, (int *)&orb_indices1[samp_idx][1]);
                                orb_indices1[samp_idx][0] = !ini_flag;
                                num_added++;
                            }
                        }
                        samp_idx++;
                    }
                    sol_vec.perform_add();
                    if (args.pt_weight) {
                        for (size_t ini_idx = start_idx; ini_idx < samp_idx; ini_idx++) {
                            if (orb_indices1[ini_idx][0]) {
                                sol_vec.add_ini_weight(orb_indices1[ini_idx][1], det_indices1[ini_idx], det_indices2[comp_idx[ini_idx][0]]);
                            }
                        }
                    }
                    num_added = sum_mpi(num_added, proc_rank, n_procs);
                }
            }
            size_t new_max_dets = sol_vec.max_size();
            if (new_max_dets > max_n_dets) {
                keep_exact.resize(new_max_dets, false);
                srt_arr = (size_t *)realloc(srt_arr, sizeof(size_t) * new_max_dets);
                for (; max_n_dets < new_max_dets; max_n_dets++) {
                    srt_arr[max_n_dets] = max_n_dets;
                }
            }
            
#pragma mark Perform deterministic subspace multiplication
            for (samp_idx = 0; samp_idx < determ_h_size; samp_idx++) {
                det_idx = determ_from[samp_idx];
                double mat_vec = vals_before_mult[det_idx] * determ_matr_el[samp_idx];
                sol_vec.add(determ_to[samp_idx], mat_vec, 1);
            }
            sol_vec.perform_add();
            
#pragma mark Death/cloning step
            sol_vec.set_curr_vec_idx(0);
            for (det_idx = 0; det_idx < vec_size; det_idx++) {
                double *curr_val = sol_vec[det_idx];
                if (*curr_val != 0) {
                    double diag_el = sol_vec.matr_el_at_pos(det_idx);
                    double pt_corr = 0;
                    if (glob_norm >= target_norm) {
                        pt_corr = args.pt_weight * (1 - sol_vec.get_pacc(det_idx));
                    }
                    *curr_val *= 1 - eps * (diag_el - en_shift + pt_corr);
                }
            }
            sol_vec.add_vecs(0, 1);
            
#pragma mark Vector compression step
            unsigned int n_samp = args.target_nonz;
            loc_norms[proc_rank] = find_preserve(&(sol_vec.values()[n_determ]), srt_arr, keep_exact, sol_vec.curr_size() - n_determ, &n_samp, &glob_norm);
            glob_norm += sol_vec.dense_norm();
            if (proc_rank == hf_proc) {
                fprintf(nkept_file, "%u\n", args.target_nonz - n_samp);
            }
            
            // Adjust shift
            if ((iterat + 1) % shift_interval == 0) {
                adjust_shift(&en_shift, glob_norm, &last_one_norm, target_norm, shift_damping / shift_interval / eps);
                if (proc_rank == hf_proc) {
                    fprintf(shift_file, "%lf\n", en_shift);
                    fprintf(norm_file, "%lf\n", glob_norm);
                }
            }
            double numer = sol_vec.dot(htrial_vec.indices(), htrial_vec.values(), htrial_vec.curr_size(), htrial_hashes);
            double denom = sol_vec.dot(trial_vec.indices(), trial_vec.values(), trial_vec.curr_size(), trial_hashes);
            numer = sum_mpi(numer, proc_rank, n_procs);
            denom = sum_mpi(denom, proc_rank, n_procs);
            if (proc_rank == hf_proc) {
                fprintf(num_file, "%lf\n", numer);
                fprintf(den_file, "%lf\n", denom);
                printf("%6u, en est: %.9lf, shift: %lf, norm: %lf\n", iterat, numer / denom, en_shift, glob_norm);
                fprintf(ini_file, "%zu\n", n_ini);
            }
            
            if (proc_rank == 0) {
                rn_sys = genrand_mt(rngen_ptr) / (1. + UINT32_MAX);
            }
#ifdef USE_MPI
            MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, loc_norms, 1, MPI_DOUBLE, MPI_COMM_WORLD);
#endif
            sys_comp(&(sol_vec.values()[n_determ]), sol_vec.curr_size() - n_determ, loc_norms, n_samp, keep_exact, rn_sys);
            for (det_idx = 0; det_idx < sol_vec.curr_size() - n_determ; det_idx++) {
                if (keep_exact[det_idx]) {
                    sol_vec.del_at_pos(det_idx + n_determ);
                    keep_exact[det_idx] = 0;
                }
            }
            
            if ((iterat + 1) % save_interval == 0) {
                sol_vec.save(result_dir);
                uint64_t tot_add = sol_vec.tot_sgn_coh();
                if (proc_rank == hf_proc) {
                    fflush(num_file);
                    fflush(den_file);
                    fflush(shift_file);
                    fflush(nkept_file);
                    printf("Total additions to nonzero: %" PRIu64 "\n", tot_add);
                }
            }
        }
        sol_vec.save(result_dir);
        if (proc_rank == hf_proc) {
            fclose(num_file);
            fclose(den_file);
            fclose(shift_file);
            fclose(nkept_file);
        }
#ifdef USE_MPI
        MPI_Finalize();
#endif
    } catch (std::exception &ex) {
        std::cerr << "\nException : " << ex.what() << "\n\nPlease report this error to the developers through our GitHub repository: https://github.com/sgreene8/FRIES/ \n\n";
    }
    return 0;
}

