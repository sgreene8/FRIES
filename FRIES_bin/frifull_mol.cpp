/*! \file
 *
 * \brief FRI algorithm with systematic matrix compression for a molecular
 * system
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <math.h>
#include <cinttypes>
#include <FRIES/io_utils.hpp>
#include <FRIES/compress_utils.hpp>
#include <FRIES/Ext_Libs/argparse.hpp>
#include <FRIES/Hamiltonians/molecule.hpp>
#include <stdexcept>

struct MyArgs : public argparse::Args {
    std::string hf_path = kwarg("hf_path", "Path to the directory that contains the HF output files eris.txt, hcore.txt, symm.txt, hf_en.txt, and sys_params.txt");
    double target_norm = kwarg("target", "Target one-norm of solution vector").set_default(0);
    uint32_t max_iter = kwarg("max_iter", "Maximum number of iterations to run the calculation").set_default(1000000);
    uint32_t target_nonz = kwarg("vec_nonz", "Target number of nonzero vector elements to keep after each iteration");
    std::string result_dir = kwarg("result_dir", "Directory in which to save output files").set_default<std::string>("./");
    uint32_t max_n_dets = kwarg("max_dets", "Maximum number of determinants on a single MPI process");
    std::shared_ptr<std::string> load_dir = kwarg("load_dir", "Directory from which to load checkpoint files from a previous FRI calculation (in binary format, see documentation for DistVec::save() and DistVec::load())");
    std::shared_ptr<std::string> ini_path = kwarg("ini_vec", "Prefix for files containing the vector with which to initialize the calculation (files must have names <ini_vec>dets and <ini_vec>vals and be text files)");
    std::shared_ptr<std::string> trial_path = kwarg("trial_vec", "Prefix for files containing the vector with which to calculate the energy (files must have names <trial_vec>dets and <trial_vec>vals and be text)");
    std::shared_ptr<std::string> rdm_path = kwarg("rdm_path", "Path to file from which to load the diagonal elements of the 1-RDM to use in compression");
    double rdm_confidence = kwarg("kernel_confidence", "Parameter to use in the kernel function when using the 1-RDM to guide compression").set_default(0);
    
    CONSTRUCTOR(MyArgs);
};

int main(int argc, char * argv[]) {
    MyArgs args(argc, argv);
    
    double target_norm = args.target_norm;
    
    try {
        int n_procs = 1;
        int proc_rank = 0;
        unsigned int hf_proc;

        MPI_Init(NULL, NULL);
        MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
        
        // Parameters
        double shift_damping = 0.05;
        unsigned int shift_interval = 10;
        unsigned int save_interval = 100;
        double en_shift = 0;
        
        // Read in data files
        hf_input in_data;
        parse_hf_input(args.hf_path, &in_data);
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
        
        std::vector<double> rdm_diag(tot_orb);
        if (args.rdm_path != nullptr) {
            load_rdm(args.rdm_path->c_str(), rdm_diag.data());
            std::copy(rdm_diag.begin() + n_frz / 2, rdm_diag.end(), rdm_diag.begin());
        }
        
        // Rn generator
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::mt19937 mt_obj((unsigned int)seed);
        
        // Solution vector
        unsigned int num_ex = n_elec_unf * n_elec_unf * (n_orb - n_elec_unf / 2) * (n_orb - n_elec_unf / 2);
        unsigned int spawn_len = args.target_nonz / n_procs * num_ex / n_procs / 4;
        size_t adder_size = spawn_len > 1000000 ? 1000000 : spawn_len;
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
                    proc_scrambler[det_idx] = mt_obj();
                }
                save_proc_hash(args.result_dir.c_str(), proc_scrambler.data(), 2 * n_orb);
            }

            MPI_Bcast(proc_scrambler.data(), 2 * n_orb, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        }
        std::vector<uint32_t> vec_scrambler(2 * n_orb);
        for (det_idx = 0; det_idx < 2 * n_orb; det_idx++) {
            vec_scrambler[det_idx] = mt_obj();
        }
        DistVec<double> sol_vec(args.max_n_dets, adder_size, n_orb * 2, n_elec_unf, n_procs, diag_shortcut, NULL, 2, proc_scrambler, vec_scrambler);
        
        std::function<void(size_t, double *)> rdm_obs = [n_elec_unf, &sol_vec, n_orb](size_t idx, double *obs_vals) {
            uint8_t *orbs = sol_vec.orbs_at_pos(idx);
            for (size_t elec_idx = 0; elec_idx < n_elec_unf; elec_idx++) {
                obs_vals[orbs[elec_idx] % n_orb] += 1;
            }
        };
        
        uint8_t hf_det[det_size];
        gen_hf_bitstring(n_orb, n_elec - n_frz, hf_det);
        hf_proc = sol_vec.idx_to_proc(hf_det);
        
        uint8_t tmp_orbs[n_elec_unf];
        uint8_t *orb_indices = (uint8_t *)malloc(sizeof(char) * 4 * num_ex);
        
# pragma mark Set up trial vector
        size_t n_trial;
        size_t n_ex = n_orb * n_orb * n_elec_unf * n_elec_unf;
        Matrix<uint8_t> &load_dets = sol_vec.indices();
        double *load_vals = (double *)sol_vec.values();
        if (args.trial_path != nullptr) { // load trial vector from file
            n_trial = load_vec_txt(*args.trial_path, load_dets, load_vals);
        }
        else {
            n_trial = 1;
        }
        DistVec<double> trial_vec(n_trial, n_trial, n_orb * 2, n_elec_unf, n_procs, proc_scrambler, vec_scrambler);
        DistVec<double> htrial_vec(n_trial * n_ex / n_procs, n_trial * n_ex / n_procs, n_orb * 2, n_elec_unf, n_procs, diag_shortcut, NULL, 2, proc_scrambler, vec_scrambler);
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
        std::vector<uintmax_t> trial_hashes(trial_vec.curr_size());
        for (det_idx = 0; det_idx < trial_vec.curr_size(); det_idx++) {
            trial_hashes[det_idx] = sol_vec.idx_to_hash(trial_vec.indices()[det_idx], tmp_orbs);
        }
        
        // Calculate H * trial vector, and accumulate results on each processor
        h_op_offdiag(htrial_vec, symm, tot_orb, *eris, *h_core, orb_indices, n_frz, n_elec_unf, 1, 1);
        htrial_vec.set_curr_vec_idx(0);
        h_op_diag(htrial_vec, 0, 0, 1);
        htrial_vec.add_vecs(0, 1);
        
        htrial_vec.collect_procs();
        std::vector<uintmax_t> htrial_hashes(htrial_vec.curr_size());
        for (det_idx = 0; det_idx < htrial_vec.curr_size(); det_idx++) {
            htrial_hashes[det_idx] = sol_vec.idx_to_hash(htrial_vec.indices()[det_idx], tmp_orbs);
        }
        
        char file_path[300];
        FILE *num_file = NULL;
        FILE *den_file = NULL;
        FILE *shift_file = NULL;
        FILE *norm_file = NULL;
        FILE *nkept_file = NULL;
        FILE *prob_file = NULL;
        
#pragma mark Initialize solution vector
        if (args.load_dir != nullptr) {
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
            Matrix<uint8_t> load_dets(args.max_n_dets, det_size);
            double *load_vals = (double *)sol_vec.values();
            
            size_t n_dets = load_vec_txt(*args.ini_path, load_dets, load_vals);
            
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
            strcpy(file_path, args.result_dir.c_str());
            strcat(file_path, "projnum.txt");
            num_file = fopen(file_path, "a");
            if (!num_file) {
                fprintf(stderr, "Could not open file for writing in directory %s\n", args.result_dir.c_str());
            }
            strcpy(file_path, args.result_dir.c_str());
            strcat(file_path, "projden.txt");
            den_file = fopen(file_path, "a");
            strcpy(file_path, args.result_dir.c_str());
            strcat(file_path, "S.txt");
            shift_file = fopen(file_path, "a");
            strcpy(file_path, args.result_dir.c_str());
            strcat(file_path, "norm.txt");
            norm_file = fopen(file_path, "a");
            strcpy(file_path, args.result_dir.c_str());
            strcat(file_path, "nkept.txt");
            nkept_file = fopen(file_path, "a");
            strcpy(file_path, args.result_dir.c_str());
            strcat(file_path, "probs.txt");
            prob_file = fopen(file_path, "a");
            
            strcpy(file_path, args.result_dir.c_str());
            strcat(file_path, "params.txt");
            FILE *param_f = fopen(file_path, "w");
            fprintf(param_f, "FRI calculation\nHF path: %s\nepsilon (imaginary time step): %lf\nTarget norm %lf\nVector nonzero: %u\n", args.hf_path.c_str(), eps, target_norm, args.target_nonz);
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
        
        double last_one_norm = 0;
        
        // Parameters for systematic sampling
        double rn_sys = 0;
        double loc_norms[n_procs];
        size_t max_n_dets = sol_vec.max_size();
        std::vector<size_t> srt_arr(max_n_dets);
        for (det_idx = 0; det_idx < max_n_dets; det_idx++) {
            srt_arr[det_idx] = det_idx;
        }
        std::vector<bool> keep_exact(max_n_dets, false);
        size_t num_rn_obs = (args.rdm_path != nullptr) ? 5 : 0;
        Matrix<double> obs_vals(num_rn_obs, n_orb);
        std::vector<double> obs_probs(num_rn_obs);
        
        for (unsigned int iterat = 0; iterat < args.max_iter; iterat++) {
            h_op_offdiag(sol_vec, symm, tot_orb, *eris, *h_core, orb_indices, n_frz, n_elec_unf, 1, -eps);
            sol_vec.set_curr_vec_idx(0);
            h_op_diag(sol_vec, 0, 1 + eps * en_shift, -eps);
            sol_vec.add_vecs(0, 1);
            
            size_t new_max_dets = sol_vec.max_size();
            if (new_max_dets > max_n_dets) {
                keep_exact.resize(new_max_dets, false);
                srt_arr.resize(new_max_dets);
                for (; max_n_dets < new_max_dets; max_n_dets++) {
                    srt_arr[max_n_dets] = max_n_dets;
                }
            }
            
#pragma mark Vector compression step
            unsigned int n_samp = args.target_nonz;
            loc_norms[proc_rank] = find_preserve(sol_vec.values(), srt_arr, keep_exact, sol_vec.curr_size(), &n_samp, &glob_norm);

            MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, loc_norms, 1, MPI_DOUBLE, MPI_COMM_WORLD);
            glob_norm += sol_vec.dense_norm();
            if (proc_rank == hf_proc) {
                fprintf(nkept_file, "%u\n", args.target_nonz - n_samp);
            }
            if (args.rdm_path != nullptr) {
                sys_obs(sol_vec.values(), sol_vec.curr_size(), loc_norms, n_samp, keep_exact, rdm_obs, obs_vals);
                double two_norm = sol_vec.two_norm();
                two_norm = sum_mpi(two_norm, proc_rank, n_procs);
                double prob_norm = 0;
                for (size_t rn_idx = 0; rn_idx < num_rn_obs; rn_idx++) {
                    obs_probs[rn_idx] = 0;
                    for (size_t obs_idx = 0; obs_idx < n_orb; obs_idx++) {
                        double obs = sum_mpi(obs_vals(rn_idx, obs_idx), proc_rank, n_procs) / two_norm;
                        obs_probs[rn_idx] += (obs - rdm_diag[obs_idx]) * (obs - rdm_diag[obs_idx]);
                    }
                    obs_probs[rn_idx] = exp(-obs_probs[rn_idx] / args.rdm_confidence / args.rdm_confidence);
                    prob_norm += obs_probs[rn_idx];
                }
                for (size_t rn_idx = 0; rn_idx < num_rn_obs; rn_idx++) {
                    obs_probs[rn_idx] /= prob_norm;
                }
                if ((iterat + 1) % 1000 == 0 && proc_rank == hf_proc) {
                    for (size_t rn_idx = 0; rn_idx < num_rn_obs; rn_idx++) {
                        fprintf(prob_file, "%lf,", obs_probs[rn_idx]);
                    }
                    fprintf(prob_file, "\n");
                }
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
            }
            
            if (proc_rank == 0) {
                rn_sys = mt_obj() / (1. + UINT32_MAX);
            }
            if (args.rdm_path != nullptr) {
                sys_comp_nonuni(sol_vec.values(), sol_vec.curr_size(), loc_norms, n_samp, keep_exact, obs_probs.data(), num_rn_obs, rn_sys);
            }
            else {
                sys_comp(sol_vec.values(), sol_vec.curr_size(), loc_norms, n_samp, keep_exact, rn_sys);
            }
            for (det_idx = 0; det_idx < sol_vec.curr_size(); det_idx++) {
                if (keep_exact[det_idx]) {
                    sol_vec.del_at_pos(det_idx);
                    keep_exact[det_idx] = 0;
                }
            }
            
            if ((iterat + 1) % save_interval == 0) {
                sol_vec.save(args.result_dir.c_str());
                if (proc_rank == hf_proc) {
                    fflush(num_file);
                    fflush(den_file);
                    fflush(shift_file);
                    fflush(nkept_file);
                    fflush(prob_file);
                }
            }
        }
        sol_vec.save(args.result_dir.c_str());
        if (proc_rank == hf_proc) {
            fclose(num_file);
            fclose(den_file);
            fclose(shift_file);
            fclose(nkept_file);
            fclose(prob_file);
        }

        MPI_Finalize();
    } catch (std::exception &ex) {
        std::cerr << "\nException : " << ex.what() << "\n\nPlease report this error to the developers through our GitHub repository: https://github.com/sgreene8/FRIES/ \n\n";
    }
    return 0;
}

