/*! \file
 *
 * \brief FRI applied to subspace iteration for calculating excited-state energies
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
#include <algorithm>

struct MyArgs : public argparse::Args {
    std::string hf_path = kwarg("hf_path", "Path to the directory that contains the HF output files eris.txt, hcore.txt, symm.txt, hf_en.txt, and sys_params.txt");
    uint32_t max_iter = kwarg("max_iter", "Maximum number of iterations to run the calculation").set_default(1000000);
    uint32_t target_nonz = kwarg("vec_nonz", "Target number of nonzero vector elements to keep after each iteration");
    std::string result_dir = kwarg("result_dir", "Directory in which to save output files").set_default<std::string>("./");
    uint32_t max_n_dets = kwarg("max_dets", "Maximum number of determinants on a single MPI process");
    std::string trial_path = kwarg("trial_vecs", "Prefix for files containing the vectors with which to calculate the energy and initialize the calculation. Files must have names <trial_vecs>dets<xx> and <trial_vecs>vals<xx>, where xx is a 2-digit number ranging from 0 to (num_trial - 1), and be text files");
    uint32_t n_trial = kwarg("num_trial", "Number of trial vectors to use to calculate dot products with the iterates");
    uint32_t restart_int = kwarg("restart_int", "Number of multiplications by (1 - \eps H) to do in between restarts").set_default(10);
    std::string mat_output = kwarg("out_format", "A flag controlling the format for outputting the Hamiltonian and overlap matrices. Must be either 'none', in which case these matrices are not written to disk, 'txt', in which case they are outputted in text format, 'npy', in which case they are outputted in numpy format, or 'bin' for binary format").set_default<std::string>("none");
    std::string normalization_technique = kwarg("norm_technique", "Specify how to normalize the iterates, i.e. not at all ('none'), individually by one-norm ('1-norm'), by the max of all 1-norms ('max-1-norm'), using energy shifts like in DMC ('shifts'), or using our new energy shift exp(S) technique ('new-shifts')");
    
    CONSTRUCTOR(MyArgs);
};

int main(int argc, char * argv[]) {
    MyArgs args(argc, argv);
    
    uint8_t n_trial = args.n_trial;

    enum {
        no_out,
        txt_out,
        npy_out,
        bin_out
    } mat_fmt;
    
    enum {
        none,
        one_norm,
        max_1_norm,
        shifts,
        new_shifts
    } norm_technique;
    
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
        else if (args.mat_output == "bin") {
            mat_fmt = bin_out;
        }
        else {
            throw std::runtime_error("out_format input argument must be \"none\", \"txt\", \"npy\", or \"bin\"");
        }
        
        if (args.normalization_technique == "none") {
            norm_technique = none;
        }
        else if (args.normalization_technique == "1-norm") {
            norm_technique = one_norm;
        }
        else if (args.normalization_technique == "max-1-norm") {
            norm_technique = max_1_norm;
        }
        else if (args.normalization_technique == "shifts") {
            norm_technique = shifts;
        }
        else if (args.normalization_technique == "new-shifts") {
            norm_technique = new_shifts;
        }
        else {
            throw std::runtime_error("norm_technique input argument must be \"none\", \"1-norm\", \"max-1-norm\", \"shifts\", \"new-shifts\"");
        }
        
        if (n_trial < 2) {
            std::cerr << "Warning: Only 1 or 0 trial vectors were provided. Consider using the power method instead of subspace iteration in this case.\n";
        }
    } catch (std::exception &ex) {
        std::cerr << "\nError parsing command line: " << ex.what() << "\n\n";
        return 1;
    }
    
    try {
        int n_procs = 1;
        int proc_rank = 0;
        
        MPI_Init(NULL, NULL);
        MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
        
        // Parameters
//        double shift_damping = 0.05;
//        unsigned int shift_interval = 10;
        
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
        
        // Parameters
        unsigned int shift_interval = 1;
        double shift_damping = eps * shift_interval * 1;
        
        // Rn generator
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::cout << "seed on process " << proc_rank << " is " << seed << std::endl;
        std::mt19937 mt_obj((unsigned int)seed);
        
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
                proc_scrambler[det_idx] = mt_obj();
            }
            save_proc_hash(args.result_dir, proc_scrambler.data(), 2 * n_orb);
        }

        MPI_Bcast(proc_scrambler.data(), 2 * n_orb, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        
        std::vector<uint32_t> vec_scrambler(2 * n_orb);
        for (size_t det_idx = 0; det_idx < 2 * n_orb; det_idx++) {
            vec_scrambler[det_idx] = mt_obj();
        }
        
        Adder<double> shared_adder(adder_size, n_procs, n_orb * 2);
        
        DistVec<double> sol_vec(args.max_n_dets, &shared_adder, n_orb * 2, n_elec_unf, diag_shortcut, 2 * n_trial, proc_scrambler, vec_scrambler);
        size_t det_size = CEILING(2 * n_orb, 8);
        
        uint8_t (*orb_indices1)[4] = (uint8_t (*)[4])malloc(sizeof(char) * 4 * num_ex);
        
# pragma mark Set up trial vectors
        std::vector<DistVec<double>> trial_vecs;
        trial_vecs.reserve(n_trial);
        
        Matrix<uint8_t> *load_dets = new Matrix<uint8_t>(args.max_n_dets, det_size);
        sol_vec.set_curr_vec_idx(n_trial);
        double *load_vals = sol_vec.values();
        std::stringstream tmp_path;
        for (uint8_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
            tmp_path << args.trial_path << std::setfill('0') << std::setw(2) << (int) trial_idx;

            unsigned int loc_n_dets = (unsigned int) load_vec_txt(tmp_path.str(), *load_dets, load_vals);
            size_t glob_n_dets = loc_n_dets;

            MPI_Bcast(&glob_n_dets, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
            trial_vecs.emplace_back(glob_n_dets, &shared_adder, n_orb * 2, n_elec_unf, proc_scrambler, vec_scrambler);
            
            for (size_t det_idx = 0; det_idx < loc_n_dets; det_idx++) {
                trial_vecs[trial_idx].add((*load_dets)[det_idx], load_vals[det_idx], 1);
            }
            trial_vecs[trial_idx].perform_add();
            sol_vec.set_curr_vec_idx(trial_idx);
            for (size_t det_idx = 0; det_idx < loc_n_dets; det_idx++) {
                sol_vec.add((*load_dets)[det_idx], load_vals[det_idx], 1);
            }
            sol_vec.perform_add();
            tmp_path.str("");
        }
        delete load_dets;
        sol_vec.fix_min_del_idx();
        
        std::vector<std::vector<uintmax_t>> trial_hashes(n_trial);
        for (uint8_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
            uint8_t tmp_orbs[n_elec_unf];
            DistVec<double> &curr_trial = trial_vecs[trial_idx];
            curr_trial.collect_procs();
            trial_hashes[trial_idx].reserve(curr_trial.curr_size());
            for (size_t det_idx = 0; det_idx < curr_trial.curr_size(); det_idx++) {
                trial_hashes[trial_idx][det_idx] = sol_vec.idx_to_hash(curr_trial.indices()[det_idx], tmp_orbs);
            }
        }
        
        std::string file_path;
        std::ofstream bmat_file;
        std::ofstream dmat_file;
        
        if (proc_rank == 0) {
            if (mat_fmt == txt_out) {
                tmp_path << args.result_dir << "b_mat.txt";
                bmat_file.open(tmp_path.str(), std::ios::app);
                if (!bmat_file.is_open()) {
                    std::string msg("Could not open file for writing in directory ");
                    msg.append(args.result_dir);
                    throw std::runtime_error(msg);
                }
                bmat_file << std::setprecision(14);
                
                tmp_path.str("");
                tmp_path << args.result_dir << "d_mat.txt";
                dmat_file.open(tmp_path.str(), std::ios::app);
                dmat_file << std::setprecision(14);
            }
            else if (mat_fmt == bin_out) {
                tmp_path << args.result_dir << "b_mat.dat";
                bmat_file.open(tmp_path.str(), std::ios::app | std::ios::binary);
                if (!bmat_file.is_open()) {
                    std::string msg("Could not open file for writing in directory ");
                    msg.append(args.result_dir);
                    throw std::runtime_error(msg);
                }
                
                tmp_path.str("");
                tmp_path << args.result_dir << "d_mat.dat";
                dmat_file.open(tmp_path.str(), std::ios::app | std::ios::binary);
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
            param_f << "Subspace iteration\nHF path: " << args.hf_path << "\nepsilon (imaginary time step): " << eps << "\nVector nonzero: " << args.target_nonz << "\n";
            param_f << "Path for trial vectors: " << args.trial_path << "\n";
            param_f << "Restart interval: " << args.restart_int << "\n";
            param_f.close();
        }
        
        // Vectors for systematic sampling
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
        tmp_path << args.result_dir << "d_mat.npy";
        std::string dnpy_path(tmp_path.str());
        tmp_path.str("");
        tmp_path << args.result_dir << "b_mat.npy";
        std::string bnpy_path(tmp_path.str());
        tmp_path.str("");
        tmp_path << args.result_dir << "shifts.txt";
        std::ofstream shift_f(tmp_path.str());
        
        std::vector<double> lapack_scratch((3 + n_trial + 32 * 2) * n_trial - 1);
        Matrix<double> lapack_inout(n_trial, n_trial);
        std::vector<double> norms(n_trial);
        std::vector<double> last_norms(n_trial);
        
        int vec_half = 0; // controls whether the current iterates are stored in the first or second half of the values_ matrix
        
        std::vector<double> en_shifts(n_trial);
        
        for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
            sol_vec.set_curr_vec_idx(vec_half * n_trial + vec_idx);
            double norm = sol_vec.local_norm();
            norm = sum_mpi(norm, proc_rank, n_procs, MPI_COMM_WORLD);
            for (size_t el_idx = 0; el_idx < sol_vec.curr_size(); el_idx++) {
                *sol_vec[el_idx] /= norm;
            }
        }
        
        for (uint32_t iteration = 0; iteration < args.max_iter; iteration++) {
            if (proc_rank == 0) {
                std::cout << "Iteration " << iteration << "\n";
            }
#pragma mark Normalize vectors
            for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                sol_vec.set_curr_vec_idx(vec_half * n_trial + vec_idx);
                double norm = sol_vec.local_norm();
                norms[vec_idx] = sum_mpi(norm, proc_rank, n_procs, MPI_COMM_WORLD);
            }
            
            if (iteration == 0) {
                std::copy(norms.begin(), norms.end(), last_norms.begin());
                if (norm_technique == new_shifts) {
                    std::fill(en_shifts.begin(), en_shifts.end(), 1);
                }
            }
            if ((iteration + 1) % shift_interval == 0) {
                if (norm_technique == shifts) {
                    for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                        adjust_shift(&en_shifts[vec_idx], norms[vec_idx], &last_norms[vec_idx], 0, shift_damping / shift_interval / eps);
                        if (proc_rank == 0) {
                            shift_f << en_shifts[vec_idx];
                            if (vec_idx + 1 < n_trial) {
                                shift_f << ",";
                            }
                        }
                    }
                }
                if (norm_technique == new_shifts) {
                    for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                        adjust_shift2(&en_shifts[vec_idx], norms[vec_idx], &last_norms[vec_idx], shift_damping / shift_interval / eps);
                        if (proc_rank == 0) {
                            shift_f << en_shifts[vec_idx];
                            if (vec_idx + 1 < n_trial) {
                                shift_f << ",";
                            }
                        }
                    }
                }
            }
            if (norm_technique == max_1_norm) {
                double max_norm = 0;
                for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                    if (norms[vec_idx] > max_norm) {
                        max_norm = norms[vec_idx];
                    }
                }
                for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                    norms[vec_idx] = max_norm;
                }
            }
            else if (norm_technique == none || norm_technique == shifts) {
                for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                    norms[vec_idx] = 1;
                }
            }
            else if (norm_technique == new_shifts) {
                for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                    norms[vec_idx] = en_shifts[vec_idx];
                }
            }
            for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                sol_vec.set_curr_vec_idx(vec_half * n_trial + vec_idx);
                for (size_t el_idx = 0; el_idx < sol_vec.curr_size(); el_idx++) {
                    *sol_vec[el_idx] /= norms[vec_idx];
                }
            }
            
#pragma mark Calculate overlap matrix
            for (uint16_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
                DistVec<double> &curr_trial = trial_vecs[trial_idx];
                for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                    sol_vec.set_curr_vec_idx(vec_half * n_trial + vec_idx);
                    double d_prod = sol_vec.dot(curr_trial.indices(), curr_trial.values(), curr_trial.curr_size(), trial_hashes[trial_idx]);
                    d_mat(trial_idx, vec_idx) = sum_mpi(d_prod, proc_rank, n_procs, MPI_COMM_WORLD);
                }
            }
            
#pragma mark Vector compression
            compress_vecs(sol_vec, vec_half * n_trial, (vec_half + 1) * n_trial, args.target_nonz, srt_arr, keep_exact, del_all, mt_obj);
            
            int use_shifts = norm_technique == shifts ? 1 : 0;
                        
# pragma mark Matrix multiplication
            for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                int curr_idx = vec_half * n_trial + vec_idx;
                int next_idx = (1 - vec_half) * n_trial + vec_idx;
                sol_vec.set_curr_vec_idx(curr_idx);
                h_op_diag(sol_vec, next_idx, 1 + eps * en_shifts[vec_idx] * use_shifts, -eps);
                sol_vec.set_curr_vec_idx(curr_idx);
                h_op_offdiag(sol_vec, symm, tot_orb, *eris, *h_core, (uint8_t *)orb_indices1, n_frz, n_elec_unf, next_idx, -eps);
            }
            vec_half = !vec_half;
            
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

#pragma mark Calculate hamiltonian matrix
            for (uint16_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
                DistVec<double> &curr_trial = trial_vecs[trial_idx];
                for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                    sol_vec.set_curr_vec_idx(vec_half * n_trial + vec_idx);
                    double d_prod = sol_vec.dot(curr_trial.indices(), curr_trial.values(), curr_trial.curr_size(), trial_hashes[trial_idx]);
                    b_mat(trial_idx, vec_idx) = sum_mpi(d_prod, proc_rank, n_procs, MPI_COMM_WORLD);
                    if (trial_idx == vec_idx) {
                        b_mat(trial_idx, trial_idx) -= eps * en_shifts[trial_idx] * use_shifts * d_mat(trial_idx, trial_idx);
                    }
                }
            }
            
            if (iteration == 0 && norm_technique == shifts) {
                for (uint16_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
                    en_shifts[trial_idx] = (d_mat(trial_idx, trial_idx) - b_mat(trial_idx, trial_idx)) / eps / d_mat(trial_idx, trial_idx);
                }
            }
            
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
                else if (mat_fmt == bin_out) {
                    bmat_file.write((char *) b_mat.data(), sizeof(double) * n_trial * n_trial);
                    dmat_file.write((char *) d_mat.data(), sizeof(double) * n_trial * n_trial);
                    bmat_file.flush();
                    dmat_file.flush();
                }
                if (norm_technique == shifts || norm_technique == new_shifts) {
                    shift_f << '\n';
                    shift_f.flush();
                }
            }
            
            if ((iteration + 1) % args.restart_int == 0) {
                lapack_inout.copy_from(b_mat);
                invr_inplace(lapack_inout, lapack_scratch.data());
                if (norm_technique == shifts || norm_technique == new_shifts) {
                    for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                        sol_vec.set_curr_vec_idx(vec_half * n_trial + vec_idx);
                        double norm = sol_vec.local_norm();
                        norms[vec_idx] = sum_mpi(norm, proc_rank, n_procs, MPI_COMM_WORLD);
                    }
                }
                for (uint8_t eigen_idx = 0; eigen_idx < n_trial; eigen_idx++) {
                    sol_vec.set_curr_vec_idx((1 - vec_half) * n_trial + eigen_idx);
                    sol_vec.zero_vec();
                    for (uint8_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                        for (size_t el_idx = 0; el_idx < sol_vec.curr_size(); el_idx++) {
                            *sol_vec((1 - vec_half) * n_trial + eigen_idx, el_idx) += *sol_vec(vec_half * n_trial + vec_idx, el_idx) * lapack_inout(vec_idx, eigen_idx);
                        }
                    }
                }
                vec_half = !vec_half;
                
                if (norm_technique == shifts || norm_technique == new_shifts) {
                    for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                        sol_vec.set_curr_vec_idx(vec_half * n_trial + vec_idx);
                        double new_norm = sol_vec.local_norm();
                        new_norm = sum_mpi(new_norm, proc_rank, n_procs, MPI_COMM_WORLD);
                        for (size_t el_idx = 0; el_idx < sol_vec.curr_size(); el_idx++) {
                            *sol_vec[el_idx] *= norms[vec_idx] / new_norm;
                        }
                    }
                }
            }
        }
        if (proc_rank == 0) {
            bmat_file.close();
            dmat_file.close();
        }
        
        MPI_Finalize();
    } catch (std::exception &ex) {
        std::cerr << "\nException : " << ex.what() << "\n\nPlease send a description of this error, a copy of the command-line arguments used, and the random number generator seeds printed for each process to the developers through our GitHub repository: https://github.com/sgreene8/FRIES/ \n\n";
    }
    return 0;
}
