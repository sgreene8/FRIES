/*! \file
 *
 * \brief FRI applied to Arnoldi method for calculating excited-state energies
 */

#include <iostream>
#include <chrono>
#include <FRIES/io_utils.hpp>
#include <FRIES/compress_utils.hpp>
#include <FRIES/Ext_Libs/argparse.hpp>
#include <FRIES/Hamiltonians/molecule.hpp>
#include <LAPACK/lapack_wrappers.hpp>
#include <FRIES/Ext_Libs/cnpy/cnpy.h>
#include <stdexcept>

struct MyArgs : public argparse::Args {
    std::string hf_path = kwarg("hf_path", "Path to the directory that contains the HF output files eris.txt, hcore.txt, symm.txt, hf_en.txt, and sys_params.txt");
    uint32_t max_iter = kwarg("max_iter", "Maximum number of iterations to run the calculation").set_default(1000000);
    uint32_t target_nonz = kwarg("vec_nonz", "Target number of nonzero vector elements to keep after each iteration");
    std::string result_dir = kwarg("result_dir", "Directory in which to save output files").set_default<std::string>("./");
    uint32_t max_n_dets = kwarg("max_dets", "Maximum number of determinants on a single MPI process");
    std::string trial_path = kwarg("trial_vecs", "Prefix for files containing the vectors with which to calculate the energy and initialize the calculation. Files must have names <trial_vecs>dets<xx> and <trial_vecs>vals<xx>, where xx is a 2-digit number ranging from 0 to (num_trial - 1), and be text files");
    uint32_t n_trial = kwarg("num_trial", "Number of trial vectors to use to calculate dot products with the iterates");
    uint32_t max_krylov = kwarg("max_krylov", "Maximum number of multiplications by (1 - \eps H) to do in between restarts").set_default(10);
    std::string mat_output = kwarg("out_format", "A flag controlling the format for outputting the Hamiltonian and overlap matrices. Must be either 'none', in which case these matrices are not written to disk, 'txt', in which case they are outputted in text format, or 'npy', in which case they are outputted in numpy format").set_default<std::string>("none");
    uint32_t n_sample = kwarg("n_sample", "The number of independent vectors to simulate in parallel").set_default(1);
    uint32_t shift_interval = kwarg("shift_int", "The interval at which to adjust the energy shifts for each vector to control normalization").set_default(10);
    
    CONSTRUCTOR(MyArgs);
};

typedef enum {
    no_out,
    txt_out,
    npy_out
} OutFmt;

int main(int argc, char * argv[]) {
    MyArgs args(argc, argv);
    
    uint8_t n_trial = args.n_trial;
    uint32_t n_sample = args.n_sample;
    OutFmt mat_fmt;
    try {
        if (args.mat_output == "none") {
            mat_fmt = no_out;
        }
        else if (args.mat_output == "txt") {
            mat_fmt = txt_out;
        }
        else if (args.mat_output == "npy") {
            mat_fmt = npy_out;
        }
        else {
            throw std::runtime_error("out_format input argument must be either \"none\", \"txt\", or \"npy\"");
        }
        if (n_trial < 2) {
            std::cerr << "Warning: Only 1 or 0 trial vectors were provided. Consider using the power method instead of Arnoldi in this case.\n";
        }
    } catch (std::exception &ex) {
        std::cerr << "\nError parsing command line: " << ex.what() << "\n\n";
        return 1;
    }
    
    try {
        int n_procs = 1;
        int proc_rank = 0;
        uint16_t procs_per_vec = 1;
        
        MPI_Init(NULL, NULL);
        MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
        
        procs_per_vec = n_procs / n_sample;
        if (procs_per_vec * n_sample != n_procs) {
            if (n_procs / procs_per_vec != n_sample) {
                n_sample = n_procs / procs_per_vec;
                std::cerr << "Increasing n_sample to " << n_sample << " in order to use more MPI processes\n";
            }
            if (n_sample * procs_per_vec != n_procs) {
                std::cerr << "Warning: only " << n_sample * procs_per_vec << " MPI processes will be used\n";
            }
        }

        int samp_idx = proc_rank / procs_per_vec;
        MPI_Comm samp_comm;
        MPI_Comm_split(MPI_COMM_WORLD, samp_idx, 0, &samp_comm);
        int samp_rank;
        MPI_Comm_rank(samp_comm, &samp_rank);
        
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
        
        // Rn generator
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::mt19937 mt_obj((unsigned int)seed);
        
        // Solution vector
        unsigned int num_ex = n_elec_unf * n_elec_unf * (n_orb - n_elec_unf / 2) * (n_orb - n_elec_unf / 2);
        unsigned int spawn_length = args.target_nonz / procs_per_vec * num_ex / procs_per_vec / 4;
        size_t adder_size = spawn_length > 1000000 ? 1000000 : spawn_length;
        std::function<double(const uint8_t *)> diag_shortcut = [tot_orb, eris, h_core, n_frz, n_elec, hf_en](const uint8_t *occ_orbs) {
            return diag_matrel(occ_orbs, tot_orb, *eris, *h_core, n_frz, n_elec) - hf_en;
        };
        
        // Initialize hash function for processors and vector
        std::vector<uint32_t> proc_scrambler(2 * n_orb);
        
        if (proc_rank == 0) {
            for (size_t det_idx = 0; det_idx < 2 * n_orb; det_idx++) {
                proc_scrambler[det_idx] = mt_obj();
            }
            save_proc_hash(args.result_dir.c_str(), proc_scrambler.data(), 2 * n_orb);
        }

        MPI_Bcast(proc_scrambler.data(), 2 * n_orb, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        
        std::vector<uint32_t> vec_scrambler(2 * n_orb);
        for (size_t det_idx = 0; det_idx < 2 * n_orb; det_idx++) {
            vec_scrambler[det_idx] = mt_obj();
        }
        
        Adder<double> shared_adder(adder_size, procs_per_vec, n_orb * 2, samp_comm);
        
        DistVec<double> sol_vec(args.max_n_dets, &shared_adder, n_orb * 2, n_elec_unf, diag_shortcut, nullptr, 2 * n_trial, proc_scrambler, vec_scrambler);
        size_t det_size = CEILING(2 * n_orb, 8);
        
        uint8_t (*orb_indices1)[4] = (uint8_t (*)[4])malloc(sizeof(char) * 4 * num_ex);
        
# pragma mark Set up trial vectors
        std::vector<DistVec<double>> trial_vecs;
        trial_vecs.reserve(n_trial);
        std::vector<DistVec<double>> htrial_vecs;
        htrial_vecs.reserve(n_trial);
        
        Matrix<uint8_t> *load_dets = new Matrix<uint8_t>(args.max_n_dets, det_size);
        sol_vec.set_curr_vec_idx(n_trial);
        double *load_vals = sol_vec.values();
        for (uint8_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
            std::stringstream vec_path;
            vec_path << args.trial_path << std::setfill('0') << std::setw(2) << (int) trial_idx;

            unsigned int loc_n_dets = (unsigned int) load_vec_txt(vec_path.str(), *load_dets, load_vals, samp_comm);
            size_t glob_n_dets = loc_n_dets;

            MPI_Bcast(&glob_n_dets, 1, MPI_UNSIGNED, 0, samp_comm);
            trial_vecs.emplace_back(glob_n_dets, &shared_adder, n_orb * 2, n_elec_unf, proc_scrambler, vec_scrambler);
            htrial_vecs.emplace_back(glob_n_dets * num_ex / n_procs, &shared_adder, n_orb * 2, n_elec_unf, diag_shortcut, (double *)NULL, 2, proc_scrambler, vec_scrambler);
            
            for (size_t det_idx = 0; det_idx < loc_n_dets; det_idx++) {
                trial_vecs[trial_idx].add(load_dets[0][det_idx], load_vals[det_idx], 1);
            }
            trial_vecs[trial_idx].perform_add();
            for (size_t det_idx = 0; det_idx < loc_n_dets; det_idx++) {
                htrial_vecs[trial_idx].add(load_dets[0][det_idx], load_vals[det_idx], 1);
            }
            htrial_vecs[trial_idx].perform_add();
            sol_vec.set_curr_vec_idx(trial_idx);
            for (size_t det_idx = 0; det_idx < loc_n_dets; det_idx++) {
                sol_vec.add(load_dets[0][det_idx], load_vals[det_idx], 1);
            }
            sol_vec.perform_add();
        }
        delete load_dets;
        sol_vec.fix_min_del_idx();
        
        std::vector<std::vector<uintmax_t>> trial_hashes(n_trial);
        std::vector<std::vector<uintmax_t>> htrial_hashes(n_trial);
        for (uint8_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
            uint8_t tmp_orbs[n_elec_unf];
            DistVec<double> &curr_trial = trial_vecs[trial_idx];
            curr_trial.collect_procs();
            trial_hashes[trial_idx].reserve(curr_trial.curr_size());
            for (size_t det_idx = 0; det_idx < curr_trial.curr_size(); det_idx++) {
                trial_hashes[trial_idx][det_idx] = sol_vec.idx_to_hash(curr_trial.indices()[det_idx], tmp_orbs);
            }
            
            DistVec<double> &curr_htrial = htrial_vecs[trial_idx];
            h_op_offdiag(curr_htrial, symm, tot_orb, *eris, *h_core, (uint8_t *)orb_indices1, n_frz, n_elec_unf, 1, 1);
            curr_htrial.set_curr_vec_idx(0);
            h_op_diag(curr_htrial, 0, 0, 1);
            curr_htrial.add_vecs(0, 1);
            curr_htrial.collect_procs();
            htrial_hashes[trial_idx].reserve(curr_htrial.curr_size());
            for (size_t det_idx = 0; det_idx < curr_htrial.curr_size(); det_idx++) {
                htrial_hashes[trial_idx][det_idx] = sol_vec.idx_to_hash(curr_htrial.indices()[det_idx], tmp_orbs);
            }
        }
        
        std::string file_path;
        std::ofstream bmat_file;
        std::ofstream dmat_file;
        std::ofstream evals_file;
//        std::ofstream restart_file;
        
        if (proc_rank == 0) {
            // Setup output files
            if (mat_fmt == txt_out) {
                file_path = args.result_dir;
                file_path.append("b_matrix.txt");
                bmat_file.open(file_path, std::ios::app);
                if (!bmat_file.is_open()) {
                    std::string msg("Could not open file for writing in directory ");
                    msg.append(args.result_dir);
                    throw std::runtime_error(msg);
                }
                bmat_file << std::setprecision(14);
                
                file_path = args.result_dir;
                file_path.append("d_matrix.txt");
                dmat_file.open(file_path, std::ios::app);
                dmat_file << std::setprecision(14);
            }
        }
        
        if (proc_rank == 0) {
            // Setup output files
            file_path = args.result_dir;
            file_path.append("params.txt");
            std::ofstream param_f(file_path);
            if (!param_f.is_open()) {
                std::string msg("Could not open file for writing in directory ");
                msg.append(args.result_dir);
                throw std::runtime_error(msg);
            }
            param_f << "Arnoldi calculation\nHF path: " << args.hf_path << "\nepsilon (imaginary time step): " << eps << "\nVector nonzero: " << args.target_nonz << "\n";
            param_f << "Path for trial vectors: " << args.trial_path << "\n";
            param_f << "Max. number of Krylov iterations: " << args.max_krylov << "\n";
            param_f.close();
            
            file_path = args.result_dir;
            file_path.append("energies.txt");
            evals_file.open(file_path);
            evals_file << std::setprecision(9);
        }
        
        // Parameters for systematic sampling
        double rn_sys = 0;
        double loc_norms[procs_per_vec];
        size_t max_n_dets = args.max_n_dets;
        if (sol_vec.max_size() > max_n_dets) {
            max_n_dets = sol_vec.max_size();
        }
        std::vector<size_t> srt_arr(max_n_dets);
        for (size_t det_idx = 0; det_idx < max_n_dets; det_idx++) {
            srt_arr[det_idx] = det_idx;
        }
        std::vector<bool> keep_exact(max_n_dets, false);
        std::vector<bool> del_all(max_n_dets, true);
        
        Matrix<double> d_mat(n_trial, n_trial);
        Matrix<double> b_mat(n_trial, n_trial);
        tmp_path.str("");
        tmp_path << args.result_dir << "d_mat_" << samp_idx << ".npy";
        std::string dnpy_path(tmp_path.str());
        tmp_path.str("");
        tmp_path << args.result_dir << "b_mat_" << samp_idx << ".npy";
        std::string bnpy_path(tmp_path.str());
        Matrix<double> d_ave(n_trial, n_trial);
        Matrix<double> b_ave(n_trial, n_trial);
        std::vector<double> s_vals(n_trial);
        std::vector<double> lapack_scratch((3 + n_trial + 32 * 2) * n_trial - 1);
        Matrix<double> evecs(n_trial, n_trial);
        std::vector<double> evals(n_trial);
        std::vector<uint8_t> ovlp_idx(n_trial);
        
        int vec_half = 0; // controls whether the current iterates are stored in the first or second half of the values_ matrix
//        uint32_t since_last_restart = 0;
        std::vector<uint8_t> eigen_sort(n_trial);
        for (uint8_t idx = 0; idx < n_trial; idx++) {
            eigen_sort[idx] = idx;
        }
        auto eigen_cmp = [&ovlp_idx](uint8_t idx1, uint8_t idx2) {
            return ovlp_idx[idx1] < ovlp_idx[idx2];
        };
        
        for (uint32_t iteration = 0; iteration < args.max_iter; iteration++) {
            if (proc_rank == 0) {
                std::cout << "Iteration " << iteration << "\n";
            }
            
#pragma mark Krylov dot products
            for (uint16_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
                DistVec<double> &curr_trial = trial_vecs[trial_idx];
                DistVec<double> &curr_htrial = htrial_vecs[trial_idx];
                for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                    sol_vec.set_curr_vec_idx(vec_half * n_trial + vec_idx);
                    double d_prod = sol_vec.dot(curr_htrial.indices(), curr_htrial.values(), curr_htrial.curr_size(), htrial_hashes[trial_idx].data());
                    b_mat(trial_idx, vec_idx) = sum_mpi(d_prod, samp_rank, procs_per_vec, samp_comm);
                    b_ave(trial_idx, vec_idx) = sum_mpi(d_prod, proc_rank, n_procs) / n_sample;
                }
            }
//            since_last_restart++;
            
            // get singular values of D
            get_svals(d_ave, s_vals, lapack_scratch.data());
            // if any are < threshold, restart
            bool restart = true; //s_vals[n_trial - 1] < 1e-6;
            if (s_vals[n_trial - 1] < 1e-8) {
                throw std::runtime_error("The overlap matrix is not full-rank");
            }
//            if (since_last_restart >= args.max_krylov) {
//                restart = true;
//            }
            
            // Get eigenvalues
            get_real_gevals_vecs(b_ave, d_ave, evals, evecs, lapack_scratch.data());
            
            // Calculate which trial vector each eigenvector overlaps with the most
            for (uint8_t eigen_idx = 0; eigen_idx < n_trial; eigen_idx++) {
                double max = 0;
                uint8_t argmax = 255;
                for (uint8_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
                    double overlap = 0;
                    for (uint8_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                        overlap += d_mat(trial_idx, vec_idx) * evecs(eigen_idx, vec_idx);
                    }
                    
                    if (fabs(overlap) > max) {
                        max = fabs(overlap);
                        argmax = trial_idx;
                    }
                }
                ovlp_idx[eigen_idx] = argmax;
            }
            
            std::sort(eigen_sort.begin(), eigen_sort.end(), eigen_cmp);
            for (uint8_t trial_idx = 0; trial_idx < n_trial - 1; trial_idx++) {
                evals_file << evals[eigen_sort[trial_idx]] << ",";
            }
            evals_file << evals[eigen_sort[n_trial - 1]] << "\n";
            evals_file.flush();
            
            if (restart) {
                if (proc_rank == 0) {
                    if (mat_fmt == npy_out) {
                        cnpy::npy_save(bnpy_path, b_mat.data(), {1, n_trial, n_trial}, "a");
                        cnpy::npy_save(dnpy_path, d_mat.data(), {1, n_trial, n_trial}, "a");
                    }
                    else if (mat_fmt == txt_out) {
                        for (uint16_t row_idx = 0; row_idx < n_trial; row_idx++) {
                            for (uint16_t col_idx = 0; col_idx < n_trial - 1; col_idx++) {
                                bmat_file << b_mat(row_idx, col_idx) << ",";
                                dmat_file << d_mat(row_idx, col_idx) << ",";
                            }
                            bmat_file << b_mat(row_idx, n_trial - 1) << "\n";
                            dmat_file << d_mat(row_idx, n_trial - 1) << "\n";
                        }
                        dmat_file.flush();
                        bmat_file.flush();
                    }
//                    restart_file << iteration << "\n";
                }
                
                // Fix signs of eigenvectors according to largest-magnitude element
                for (uint8_t eigen_idx = 0; eigen_idx < n_trial; eigen_idx++) {
                    uint8_t argmax = 0;
                    double max = fabs(evecs(eigen_idx, 0));
                    for (uint8_t vec_idx = 1; vec_idx < n_trial; vec_idx++) {
                        double el = fabs(evecs(eigen_idx, vec_idx));
                        if (el > max) {
                            max = el;
                            argmax = vec_idx;
                        }
                    }
                    int sgn = SIGN(evecs(eigen_idx, argmax));
                    for (uint8_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                        evecs(eigen_idx, vec_idx) *= sgn;
                    }
                }
                
                for (uint8_t eigen_idx = 0; eigen_idx < n_trial; eigen_idx++) {
                    sol_vec.set_curr_vec_idx((1 - vec_half) * n_trial + eigen_idx);
                    sol_vec.zero_vec();
                    for (uint8_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                        for (size_t el_idx = 0; el_idx < sol_vec.curr_size(); el_idx++) {
                            *sol_vec((1 - vec_half) * n_trial + eigen_idx, el_idx) += *sol_vec(vec_half * n_trial + vec_idx, el_idx) * evecs(eigen_sort[eigen_idx], vec_idx);
                        }
                    }
                }
                vec_half = !vec_half;
//                since_last_restart = 0;
            }
//            for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
//                sol_vecs[vec_idx].copy_vec(2, 0);
//            }
            for (uint32_t mult_idx = 0; mult_idx < args.max_krylov; mult_idx++) {
                if (proc_rank == 0) {
                    std::cout << "multiplying\n";
                }
# pragma mark Matrix multiplication
                for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                    int curr_idx = vec_half * n_trial + vec_idx;
                    int next_idx = (1 - vec_half) * n_trial + vec_idx;
                    sol_vec.set_curr_vec_idx(curr_idx);
                    h_op_offdiag(sol_vec, symm, tot_orb, *eris, *h_core, (uint8_t *)orb_indices1, n_frz, n_elec_unf, next_idx, -eps);
                    sol_vec.set_curr_vec_idx(curr_idx);
                    h_op_diag(sol_vec, curr_idx, 1 + eps * en_shift[vec_idx], -eps);
                    sol_vec.add_vecs(curr_idx, next_idx);
                }
                size_t new_max_dets = max_n_dets;
                if (sol_vec.max_size() > new_max_dets) {
                    new_max_dets = sol_vec.max_size();
                    keep_exact.resize(new_max_dets, false);
                    del_all.resize(new_max_dets, true);
                    srt_arr.resize(new_max_dets);
                    for (size_t idx = max_n_dets; idx < new_max_dets; idx++) {
                        srt_arr[idx] = idx;
                    }
                    max_n_dets = new_max_dets;
                }
#pragma mark Vector compression
                for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                    unsigned int n_samp = args.target_nonz;
                    double glob_norm;
                    sol_vec.set_curr_vec_idx(vec_half * n_trial + vec_idx);
                    loc_norms[proc_rank] = find_preserve(sol_vec.values(), srt_arr.data(), keep_exact, sol_vec.curr_size(), &n_samp, &glob_norm);
                    for (size_t el_idx = 0; el_idx < sol_vec.curr_size(); el_idx++) {
                        *sol_vec[el_idx] /= glob_norm;
                    }
//                    if ((iteration * args.max_krylov + mult_idx + 1) % shift_interval == 0) {
//                        adjust_shift(&en_shift[vec_idx], glob_norm, &last_one_norm[vec_idx], 0, shift_damping / shift_interval / eps);
//                    }
                    if (proc_rank == 0) {
                        rn_sys = mt_obj() / (1. + UINT32_MAX);
                    }

                    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, loc_norms, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                    sys_comp(sol_vec.values(), sol_vec.curr_size(), loc_norms, n_samp, keep_exact, rn_sys);
                    for (size_t det_idx = 0; det_idx < sol_vec.curr_size(); det_idx++) {
                        if (keep_exact[det_idx]) {
                            keep_exact[det_idx] = 0;
                        }
                        else {
                            del_all[det_idx] = false;
                        }
                    }
                }
                for (size_t det_idx = 0; det_idx < sol_vec.curr_size(); det_idx++) {
                    if (del_all[det_idx]) {
                        sol_vec.del_at_pos(det_idx);
                    }
                    del_all[det_idx] = true;
                }
            }
        }
        if (proc_rank == 0) {
            evals_file.close();
            bmat_file.close();
            dmat_file.close();
//            restart_file.close();
        }

        MPI_Comm_free(&samp_comm);
        MPI_Finalize();
    } catch (std::exception &ex) {
        std::cerr << "\nException : " << ex.what() << "\n\nPlease report this error to the developers through our GitHub repository: https://github.com/sgreene8/FRIES/ \n\n";
    }
    return 0;
}
