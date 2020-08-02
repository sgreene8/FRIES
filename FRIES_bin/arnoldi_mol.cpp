/*! \file
 *
 * \brief FRI applied to Arnoldi method for calculating excited-state energies
 */

#include <cstdio>
#include <iostream>
#include <ctime>
#include <FRIES/io_utils.hpp>
#include <FRIES/Ext_Libs/dcmt/dc.h>
#include <FRIES/compress_utils.hpp>
#include <FRIES/Ext_Libs/argparse.hpp>
#include <FRIES/Hamiltonians/molecule.hpp>
//#include <FRIES/Ext_Libs/LAPACK/lapacke.h>
#include <FRIES/Ext_Libs/cnpy/cnpy.h>
#include <stdexcept>

struct MyArgs : public argparse::Args {
    std::string hf_path = kwarg("hf_path", "Path to the directory that contains the HF output files eris.txt, hcore.txt, symm.txt, hf_en.txt, and sys_params.txt");
    uint32_t max_iter = kwarg("max_iter", "Maximum number of iterations to run the calculation").set_default(1000000);
    uint32_t target_nonz = kwarg("vec_nonz", "Target number of nonzero vector elements to keep after each iteration");
    std::string result_dir = kwarg("result_dir", "Directory in which to save output files").set_default<std::string>("./");
    uint32_t max_n_dets = kwarg("max_dets", "Maximum number of determinants on a single MPI process");
    std::string trial_path = kwarg("trial_vecs", "Prefix for files containing the vectors with which to calculate the energy and initialize the calculation. Files must have names <trial_vecs>dets<xx> and <trial_vecs>vals<xx>, where xx is a 2-digit number ranging from 0 to (num_trial - 1), and be text files");
    uint8_t n_trial = kwarg("num_trial", "Number of trial vectors to use to calculate dot products with the iterates");
    uint16_t n_krylov = kwarg("n_krylov", "Number of multiplications by (1 - \eps H) to include in each iteration").set_default(1000);
    bool use_npy = flag("use_numpy", "If set, output files will be in numpy (.npy) format. Otherwise, will be in text (.txt) format");
    
    CONSTRUCTOR(MyArgs);
};

int main(int argc, char * argv[]) {
    MyArgs args(argc, argv);
    
    uint8_t n_trial = args.n_trial;
    
    if (n_trial < 2) {
        fprintf(stderr, "Warning: Only 1 or 0 trial vectors were provided. Consider using the power method instead of Arnoldi in this case.\n");
    }
    
    try {
        int n_procs = 1;
        int proc_rank = 0;
#ifdef USE_MPI
        MPI_Init(NULL, NULL);
        MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
#endif
        
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
        unsigned int num_ex = n_elec_unf * n_elec_unf * (n_orb - n_elec_unf / 2) * (n_orb - n_elec_unf / 2);
        unsigned int spawn_length = args.target_nonz / n_procs * num_ex / n_procs / 4;
        size_t adder_size = spawn_length > 1000000 ? 1000000 : spawn_length;
        std::function<double(const uint8_t *)> diag_shortcut = [tot_orb, eris, h_core, n_frz, n_elec, hf_en](const uint8_t *occ_orbs) {
            return diag_matrel(occ_orbs, tot_orb, *eris, *h_core, n_frz, n_elec) - hf_en;
        };
        
        // Initialize hash function for processors and vector
        std::vector<uint32_t> proc_scrambler(2 * n_orb);
        
        if (proc_rank == 0) {
            for (size_t det_idx = 0; det_idx < 2 * n_orb; det_idx++) {
                proc_scrambler[det_idx] = genrand_mt(rngen_ptr);
            }
            save_proc_hash(args.result_dir.c_str(), proc_scrambler.data(), 2 * n_orb);
        }
#ifdef USE_MPI
        MPI_Bcast(proc_scrambler.data(), 2 * n_orb, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
#endif
        
        std::vector<uint32_t> vec_scrambler(2 * n_orb);
        for (size_t det_idx = 0; det_idx < 2 * n_orb; det_idx++) {
            vec_scrambler[det_idx] = genrand_mt(rngen_ptr);
        }
        
        std::vector<DistVec<double>> sol_vecs;
        sol_vecs.reserve(n_trial);
        for (uint8_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
            sol_vecs.emplace_back(args.max_n_dets, adder_size, n_orb * 2, n_elec_unf, n_procs, diag_shortcut, nullptr, 3, proc_scrambler, vec_scrambler);
        }
        size_t det_size = CEILING(2 * n_orb, 8);
        
        uint8_t (*orb_indices1)[4] = (uint8_t (*)[4])malloc(sizeof(char) * 4 * num_ex);
        
# pragma mark Set up trial vectors
        std::vector<DistVec<double>> trial_vecs;
        trial_vecs.reserve(n_trial);
        std::vector<DistVec<double>> htrial_vecs;
        htrial_vecs.reserve(n_trial);
        
        char vec_path[300];
        Matrix<uint8_t> *load_dets = new Matrix<uint8_t>(args.max_n_dets, det_size);
        for (unsigned int trial_idx = 0; trial_idx < n_trial; trial_idx++) {
            DistVec<double> &curr_sol = sol_vecs[trial_idx];
            double *load_vals = curr_sol.values();
            
            sprintf(vec_path, "%s%02d", args.trial_path.c_str(), trial_idx);
            unsigned int loc_n_dets = (unsigned int) load_vec_txt(vec_path, *load_dets, load_vals, DOUB);
            size_t glob_n_dets = loc_n_dets;
#ifdef USE_MPI
            MPI_Bcast(&glob_n_dets, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
#endif
            trial_vecs.emplace_back(glob_n_dets, glob_n_dets, n_orb * 2, n_elec_unf, n_procs, proc_scrambler, vec_scrambler);
            htrial_vecs.emplace_back(glob_n_dets * num_ex / n_procs, glob_n_dets * num_ex / n_procs, n_orb * 2, n_elec_unf, n_procs, diag_shortcut, (double *)NULL, 2, proc_scrambler, vec_scrambler);
            
            curr_sol.set_curr_vec_idx(2);
            for (size_t det_idx = 0; det_idx < loc_n_dets; det_idx++) {
                trial_vecs[trial_idx].add(load_dets[0][det_idx], load_vals[det_idx], 1);
                htrial_vecs[trial_idx].add(load_dets[0][det_idx], load_vals[det_idx], 1);
                curr_sol.add(load_dets[0][det_idx], load_vals[det_idx], 1);
            }
            loc_n_dets++; // just to be safe
            bzero(load_vals, loc_n_dets * sizeof(double));
            curr_sol.perform_add();
            curr_sol.fix_min_del_idx();
        }
        delete load_dets;
        
        std::vector<std::vector<uintmax_t>> trial_hashes(n_trial);
        std::vector<std::vector<uintmax_t>> htrial_hashes(n_trial);
        for (uint8_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
            uint8_t tmp_orbs[n_elec_unf];
            DistVec<double> &curr_trial = trial_vecs[trial_idx];
            curr_trial.perform_add();
            curr_trial.collect_procs();
            trial_hashes[trial_idx].reserve(curr_trial.curr_size());
            for (size_t det_idx = 0; det_idx < curr_trial.curr_size(); det_idx++) {
                trial_hashes[trial_idx][det_idx] = sol_vecs[0].idx_to_hash(curr_trial.indices()[det_idx], tmp_orbs);
            }
            
            DistVec<double> &curr_htrial = htrial_vecs[trial_idx];
            curr_htrial.perform_add();
            h_op_offdiag(curr_htrial, symm, tot_orb, *eris, *h_core, (uint8_t *)orb_indices1, n_frz, n_elec_unf, 1, 1);
            curr_htrial.set_curr_vec_idx(0);
            h_op_diag(curr_htrial, 0, 0, 1);
            curr_htrial.add_vecs(0, 1);
            curr_htrial.collect_procs();
            htrial_hashes[trial_idx].reserve(   curr_htrial.curr_size());
            for (size_t det_idx = 0; det_idx < curr_htrial.curr_size(); det_idx++) {
                htrial_hashes[trial_idx][det_idx] = sol_vecs[0].idx_to_hash(curr_htrial.indices()[det_idx], tmp_orbs);
            }
        }
        
        char file_path[300];
        FILE *bmat_file = NULL;
        FILE *dmat_file = NULL;
        
        if (proc_rank == 0) {
            // Setup output files
            if (!args.use_npy) {
                strcpy(file_path, args.result_dir.c_str());
                strcat(file_path, "b_matrix.txt");
                bmat_file = fopen(file_path, "a");
                if (!bmat_file) {
                    fprintf(stderr, "Could not open file for writing in directory %s\n", args.result_dir.c_str());
                }
                
                strcpy(file_path, args.result_dir.c_str());
                strcat(file_path, "d_matrix.txt");
                dmat_file = fopen(file_path, "a");
            }
            
            strcpy(file_path, args.result_dir.c_str());
            strcat(file_path, "params.txt");
            FILE *param_f = fopen(file_path, "w");
            fprintf(param_f, "Arnoldi calculation\nHF path: %s\nepsilon (imaginary time step): %lf\nVector nonzero: %u\n", args.hf_path.c_str(), eps, args.target_nonz);
            fprintf(param_f, "Path for trial vectors: %s\n", args.trial_path.c_str());
            fprintf(param_f, "Krylov iterations: %u\n", args.n_krylov);
            fclose(param_f);
        }
        
        // Parameters for systematic sampling
        double rn_sys = 0;
        double loc_norms[n_procs];
        size_t max_n_dets = args.max_n_dets;
        for (unsigned int vec_idx = 0; vec_idx < n_trial; vec_idx++) {
            if (sol_vecs[vec_idx].max_size() > max_n_dets) {
                max_n_dets = sol_vecs[vec_idx].max_size();
            }
        }
        std::vector<size_t> srt_arr(max_n_dets);
        for (size_t det_idx = 0; det_idx < max_n_dets; det_idx++) {
            srt_arr[det_idx] = det_idx;
        }
        std::vector<bool> keep_exact(max_n_dets, false);
        
        Matrix<double> d_mat(n_trial, n_trial);
        Matrix<double> b_mat(n_trial, n_trial);
        std::stringstream dnpy_path;
        dnpy_path << args.result_dir << "d_matrix.npy";
        std::stringstream bnpy_path;
        bnpy_path << args.result_dir << "b_matrix.npy";
        
        for (uint32_t iteration = 0; iteration < args.max_iter; iteration++) {
            // Initialize the solution vectors
            for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                sol_vecs[vec_idx].copy_vec(2, 0);
            }
            if (proc_rank == 0) {
                printf("Macro iteration %u\n", iteration);
            }
            
            for (uint16_t krylov_idx = 0; krylov_idx < args.n_krylov; krylov_idx++) {
#pragma mark Krylov dot products
                for (uint16_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
                    DistVec<double> &curr_trial = trial_vecs[trial_idx];
                    DistVec<double> &curr_htrial = htrial_vecs[trial_idx];
                    for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                        double d_prod = sol_vecs[vec_idx].dot(curr_trial.indices(), curr_trial.values(), curr_trial.curr_size(), trial_hashes[trial_idx].data());
                        d_prod = sum_mpi(d_prod, proc_rank, n_procs);
                        d_mat(trial_idx, vec_idx) = d_prod;
                        
                        d_prod = sol_vecs[vec_idx].dot(curr_htrial.indices(), curr_htrial.values(), curr_htrial.curr_size(), htrial_hashes[trial_idx].data());
                        d_prod = sum_mpi(d_prod, proc_rank, n_procs);
                        b_mat(trial_idx, vec_idx) = d_prod;
                    }
                }
                if (proc_rank == 0) {
                    if (args.use_npy) {
                        cnpy::npy_save(bnpy_path.str(), b_mat.data(), {1, n_trial, n_trial}, "a");
                        cnpy::npy_save(dnpy_path.str(), d_mat.data(), {1, n_trial, n_trial}, "a");
                    }
                    else {
                        for (uint16_t row_idx = 0; row_idx < n_trial; row_idx++) {
                            for (uint16_t col_idx = 0; col_idx < n_trial - 1; col_idx++) {
                                fprintf(bmat_file, "%.14lf,", b_mat(row_idx, col_idx));
                                fprintf(dmat_file, "%.14lf,", d_mat(row_idx, col_idx));
                            }
                            fprintf(bmat_file, "%.14lf\n", b_mat(row_idx, n_trial - 1));
                            fprintf(dmat_file, "%.14lf\n", d_mat(row_idx, n_trial - 1));
                        }
                    }
                    printf("Krylov iteration %u\n", krylov_idx);
                    fflush(dmat_file);
                    fflush(bmat_file);
                }
                
# pragma mark Matrix multiplication
                size_t new_max_dets = max_n_dets;
                for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                    DistVec<double> &curr_vec = sol_vecs[vec_idx];
                    curr_vec.set_curr_vec_idx(0);
                    h_op_offdiag(curr_vec, symm, tot_orb, *eris, *h_core, (uint8_t *)orb_indices1, n_frz, n_elec_unf, 1, -eps);
                    curr_vec.set_curr_vec_idx(0);
                    h_op_diag(curr_vec, 0, 1, -eps);
                    curr_vec.add_vecs(0, 1);
                    if (curr_vec.max_size() > new_max_dets) {
                        new_max_dets = curr_vec.max_size();
                    }
                }
                
                if (new_max_dets > max_n_dets) {
                    keep_exact.resize(new_max_dets, false);
                    srt_arr.resize(new_max_dets);
                }
#pragma mark Vector compression
                for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                    unsigned int n_samp = args.target_nonz;
                    double glob_norm;
                    for (size_t det_idx = 0; det_idx < sol_vecs[vec_idx].curr_size(); det_idx++) {
                        srt_arr[det_idx] = det_idx;
                    }
                    loc_norms[proc_rank] = find_preserve(sol_vecs[vec_idx].values(), srt_arr.data(), keep_exact, sol_vecs[vec_idx].curr_size(), &n_samp, &glob_norm);
                    if (proc_rank == 0) {
                        rn_sys = genrand_mt(rngen_ptr) / (1. + UINT32_MAX);
                    }
#ifdef USE_MPI
                    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, loc_norms, 1, MPI_DOUBLE, MPI_COMM_WORLD);
#endif
                    sys_comp(sol_vecs[vec_idx].values(), sol_vecs[vec_idx].curr_size(), loc_norms, n_samp, keep_exact, rn_sys);
                    for (size_t det_idx = 0; det_idx < sol_vecs[vec_idx].curr_size(); det_idx++) {
                        if (keep_exact[det_idx]) {
                            sol_vecs[vec_idx].del_at_pos(det_idx);
                            keep_exact[det_idx] = 0;
                        }
                    }
                }
            }
            //        LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'N', n_trial, krylov_mat.data(), n_trial, energies_r, energies_i, NULL, n_trial, NULL, n_trial);
        }
        
        if (proc_rank == 0 && !args.use_npy) {
            fclose(bmat_file);
            fclose(dmat_file);
        }
#ifdef USE_MPI
        MPI_Finalize();
#endif
    } catch (std::exception &ex) {
        std::cerr << "\nException : " << ex.what() << "\n\nPlease report this error to the developers through our GitHub repository: https://github.com/sgreene8/FRIES/ \n\n";
    }
    return 0;
}
