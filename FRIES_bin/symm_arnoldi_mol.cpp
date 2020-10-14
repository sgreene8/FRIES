/*! \file
 *
 * \brief FRI applied to Arnoldi method for calculating excited-state energies
 * Rayleigh quotients are constructed from the same random vector on the L & R sides
 */

#include <cstdio>
#include <iostream>
#include <chrono>
#include <FRIES/io_utils.hpp>
#include <FRIES/compress_utils.hpp>
#include <FRIES/Ext_Libs/argparse.hpp>
#include <FRIES/Hamiltonians/molecule.hpp>
#include <FRIES/Ext_Libs/cnpy/cnpy.h>
#include <stdexcept>

struct MyArgs : public argparse::Args {
    std::string hf_path = kwarg("hf_path", "Path to the directory that contains the HF output files eris.txt, hcore.txt, symm.txt, hf_en.txt, and sys_params.txt");
    uint32_t max_iter = kwarg("max_iter", "Maximum number of iterations to run the calculation").set_default(1000000);
    uint32_t target_nonz = kwarg("vec_nonz", "Target number of nonzero vector elements to keep after each iteration");
    std::string result_dir = kwarg("result_dir", "Directory in which to save output files").set_default<std::string>("./");
    uint32_t max_n_dets = kwarg("max_dets", "Maximum number of determinants on a single MPI process");
    std::string ini_path = kwarg("ini_vecs", "Prefix for files containing the vectors with which to initialize the calculation. Files must have names <ini_vecs>dets<xx> and <ini_vecs>vals<xx>, where xx is a 2-digit number ranging from 0 to (num_states - 1), and be text files");
    uint32_t n_states = kwarg("num_states", "Number of states whose energies will be computed");
    uint32_t max_krylov = kwarg("max_krylov", "Number of multiplications by (1 - \eps H) to do in between restarts").set_default(10);
    std::string mat_output = kwarg("out_format", "A flag controlling the format for outputting the Hamiltonian and overlap matrices. Must be either 'none', in which case these matrices are not written to disk, 'txt', in which case they are outputted in text format, or 'npy', in which case they are outputted in numpy format").set_default<std::string>("none");
    uint32_t n_sample = kwarg("n_sample", "The number of independent vectors to simulate in parallel").set_default(1);
    
    CONSTRUCTOR(MyArgs);
};

typedef enum {
    no_out,
    txt_out,
    npy_out
} OutFmt;

int main(int argc, char * argv[]) {
    MyArgs args(argc, argv);
    
    uint8_t n_states = args.n_states;
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
        if (n_states < 2) {
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
        
//        double shift_damping = 0.05;
//        uint32_t shift_interval = 10;
        std::vector<double> en_shift(n_states);
        std::vector<double> last_one_norm(n_states);
        
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
            save_proc_hash(args.result_dir.c_str(), proc_scrambler.data(), 2 * n_orb);
        }

        MPI_Bcast(proc_scrambler.data(), 2 * n_orb, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        
        std::vector<uint32_t> vec_scrambler(2 * n_orb);
        for (size_t det_idx = 0; det_idx < 2 * n_orb; det_idx++) {
            vec_scrambler[det_idx] = mt_obj();
        }
        
        Adder<double> shared_adder(adder_size, n_procs, n_orb * 2, samp_comm);
        
        std::vector<DistVec<double>> sol_vecs;
        sol_vecs.reserve(n_states);
        // vector 0: current iteration before multiplication
        // vector 1: current iteration after multiplication
        // vector 2: temporary vector for off-diagonal multiplication
        // vector 3: initial vector
        for (uint8_t vec_idx = 0; vec_idx < n_states; vec_idx++) {
            sol_vecs.emplace_back(args.max_n_dets, &shared_adder, n_orb * 2, n_elec_unf, diag_shortcut, nullptr, 4, proc_scrambler, vec_scrambler);
        }
        size_t det_size = CEILING(2 * n_orb, 8);
        
        uint8_t (*orb_indices1)[4] = (uint8_t (*)[4])malloc(sizeof(char) * 4 * spawn_length);
        
# pragma mark Set up vectors
        char vec_path[300];
        Matrix<uint8_t> *load_dets = new Matrix<uint8_t>(args.max_n_dets, det_size);
        for (unsigned int trial_idx = 0; trial_idx < n_states; trial_idx++) {
            DistVec<double> &curr_sol = sol_vecs[trial_idx];
            double *load_vals = curr_sol.values();
            
            sprintf(vec_path, "%s%02d", args.ini_path.c_str(), trial_idx);
            unsigned int loc_n_dets = (unsigned int) load_vec_txt(vec_path, *load_dets, load_vals);
            curr_sol.set_curr_vec_idx(3);
            for (size_t det_idx = 0; det_idx < loc_n_dets; det_idx++) {
                curr_sol.add(load_dets[0][det_idx], load_vals[det_idx], 1);
            }
            loc_n_dets++; // just to be safe
            bzero(load_vals, loc_n_dets * sizeof(double));
            curr_sol.perform_add();
            curr_sol.fix_min_del_idx();
        }
        delete load_dets;
        
        std::string file_path;
        std::stringstream tmp_path;
        std::ofstream bmat_file;
        std::ofstream dmat_file;
        std::ofstream evals_file;
        
        if (samp_rank == 0) {
            if (mat_fmt == txt_out) {
                tmp_path << args.result_dir << "b_mat_" << samp_idx << ".txt";
                bmat_file.open(tmp_path.str(), std::ios::app);
                if (!bmat_file.is_open()) {
                    std::string msg("Could not open file for writing in directory ");
                    msg.append(args.result_dir);
                    throw std::runtime_error(msg);
                }
                bmat_file << std::setprecision(14);
                
                tmp_path.str("");
                tmp_path << args.result_dir << "d_mat_" << samp_idx << ".txt";
                dmat_file.open(tmp_path.str(), std::ios::app);
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
            param_f << "Path for initial vectors: " << args.ini_path << "\n";
            param_f << "Max. number of Krylov iterations: " << args.max_krylov << "\n";
            param_f.close();
        }
        
        // Parameters for systematic sampling
        double rn_sys = 0;
        double loc_norms[n_procs];
        size_t max_n_dets = args.max_n_dets;
        for (unsigned int vec_idx = 0; vec_idx < n_states; vec_idx++) {
            if (sol_vecs[vec_idx].max_size() > max_n_dets) {
                max_n_dets = (unsigned int)sol_vecs[vec_idx].max_size();
            }
        }
        std::vector<size_t> srt_arr(max_n_dets);
        for (size_t det_idx = 0; det_idx < max_n_dets; det_idx++) {
            srt_arr[det_idx] = det_idx;
        }
        std::vector<bool> keep_exact(max_n_dets, false);
        
        Matrix<double> d_mat(n_states, n_states);
        Matrix<double> b_mat(n_states, n_states);
        tmp_path.str(args.result_dir);
        tmp_path << "d_mat_" << samp_idx << ".npy";
        std::string dnpy_path(tmp_path.str());
        tmp_path.str(args.result_dir);
        tmp_path << "b_mat_" << samp_idx << ".npy";
        std::string bnpy_path(tmp_path.str());
        
        for (uint32_t iteration = 0; iteration < args.max_iter; iteration++) {
            // Initialize the solution vectors
            for (uint16_t vec_idx = 0; vec_idx < n_states; vec_idx++) {
                sol_vecs[vec_idx].copy_vec(3, 0);
            }
            if (proc_rank == 0) {
                printf("Macro iteration %u\n", iteration);
            }
            
            for (uint16_t krylov_idx = 0; krylov_idx < args.max_krylov; krylov_idx++) {
                if (proc_rank == 0) {
                    printf("Krylov iteration %u\n", krylov_idx);
                }
                
# pragma mark Matrix multiplication
                for (uint16_t vec_idx = 0; vec_idx < n_states; vec_idx++) {
                    DistVec<double> &curr_vec = sol_vecs[vec_idx];
                    curr_vec.copy_vec(0, 1);
                    curr_vec.set_curr_vec_idx(1);
                    h_op_offdiag(curr_vec, symm, tot_orb, *eris, *h_core, (uint8_t *)orb_indices1, n_frz, n_elec_unf, 2, -eps);
                    curr_vec.set_curr_vec_idx(1);
                    h_op_diag(curr_vec, 1, 1 + eps * en_shift[vec_idx], -eps);
                    curr_vec.add_vecs(1, 2);
                    for (size_t det_idx = 0; det_idx < curr_vec.curr_size(); det_idx++) {
                        char det_str[2 * det_size + 1];
                        print_str(curr_vec.indices()[det_idx], det_size, det_str);
                        std::cout << det_str << ", " << curr_vec.values()[det_idx] << "\n";
                    }
                }
                
#pragma mark Krylov dot products
                for (uint16_t vecl_idx = 0; vecl_idx < n_states; vecl_idx++) {
                    DistVec<double> &lvec = sol_vecs[vecl_idx];
                    lvec.set_curr_vec_idx(0);
                    for (uint16_t vecr_idx = 0; vecr_idx < n_states; vecr_idx++) {
                        double vec_ovlp;
                        double vec_H_ovlp;
                        if (vecl_idx == vecr_idx) {
                            vec_ovlp = lvec.internal_dot(0, 0);
                            vec_ovlp = sum_mpi(vec_ovlp, proc_rank, n_procs);
                            
                            vec_H_ovlp = lvec.internal_dot(0, 1);
                            vec_H_ovlp = sum_mpi(vec_H_ovlp, proc_rank, n_procs);
                        }
                        else {
                            DistVec<double> &rvec = sol_vecs[vecr_idx];
                            rvec.set_curr_vec_idx(0);
                            vec_ovlp = lvec.multi_dot(rvec.indices(), rvec.occ_orbs(), rvec.values(), rvec.curr_size());
                            
                            rvec.set_curr_vec_idx(1);
                            vec_H_ovlp = lvec.multi_dot(rvec.indices(), rvec.occ_orbs(), rvec.values(), rvec.curr_size());
                        }
                        // <v|(1 - \eps H + \eps S)|v> = (1 + \eps S)<v|v> - \eps <v|H|v>
//                        vec_H_ovlp = -(vec_H_ovlp - (1 + eps * en_shift[vecr_idx]) * vec_ovlp) / eps; \\ here's the problem
                        b_mat(vecl_idx, vecr_idx) = vec_H_ovlp;
                        d_mat(vecl_idx, vecr_idx) = vec_ovlp;
                    }
                }
                if (samp_rank == 0) {
                    if (mat_fmt == npy_out) {
                        cnpy::npy_save(bnpy_path, b_mat.data(), {1, n_states, n_states}, "a");
                        cnpy::npy_save(dnpy_path, d_mat.data(), {1, n_states, n_states}, "a");
                    }
                    else if (mat_fmt == txt_out) {
                        for (uint16_t row_idx = 0; row_idx < n_states; row_idx++) {
                            for (uint16_t col_idx = 0; col_idx < n_states - 1; col_idx++) {
                                bmat_file << b_mat(row_idx, col_idx) << ",";
                                dmat_file << d_mat(row_idx, col_idx) << ",";
                            }
                            bmat_file << b_mat(row_idx, n_states - 1) << "\n";
                            dmat_file << d_mat(row_idx, n_states - 1) << "\n";
                        }
                        dmat_file.flush();
                        bmat_file.flush();
                    }
                }
                
                size_t new_max_dets = max_n_dets;
                for (uint16_t vec_idx = 0; vec_idx < n_states; vec_idx++) {
                    DistVec<double> &curr_vec = sol_vecs[vec_idx];
                    curr_vec.copy_vec(1, 0);
                    if (curr_vec.max_size() > new_max_dets) {
                        new_max_dets = curr_vec.max_size();
                    }
                }
                
                if (new_max_dets > max_n_dets) {
                    keep_exact.resize(new_max_dets, false);
                    srt_arr.resize(new_max_dets);
                }
#pragma mark Vector compression
                for (uint16_t vec_idx = 0; vec_idx < n_states; vec_idx++) {
                    unsigned int n_samp = args.target_nonz;
                    double glob_norm;
                    for (size_t det_idx = 0; det_idx < sol_vecs[vec_idx].curr_size(); det_idx++) {
                        srt_arr[det_idx] = det_idx;
                    }
                    loc_norms[proc_rank] = find_preserve(sol_vecs[vec_idx].values(), srt_arr, keep_exact, sol_vecs[vec_idx].curr_size(), &n_samp, &glob_norm);
                    double mean_norm = sum_mpi(glob_norm, proc_rank, n_procs) / n_procs;
//                    for (size_t el_idx = 0; el_idx < sol_vec.curr_size(); el_idx++) {
//                        *sol_vec[el_idx] /= mean_norm;
//                    }
                    
//                    if ((krylov_idx + 1) % shift_interval == 0) {
//                        adjust_shift(&en_shift[vec_idx], glob_norm, &last_one_norm[vec_idx], 0, shift_damping / shift_interval / eps);
//                    }
                    if (proc_rank == 0) {
                        rn_sys = mt_obj() / (1. + UINT32_MAX);
                    }

                    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, loc_norms, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                    sys_comp(sol_vecs[vec_idx].values(), sol_vecs[vec_idx].curr_size(), loc_norms, n_samp, keep_exact, rn_sys);
                    for (size_t det_idx = 0; det_idx < sol_vecs[vec_idx].curr_size(); det_idx++) {
                        if (keep_exact[det_idx]) {
                            sol_vecs[vec_idx].del_at_pos(det_idx);
                            keep_exact[det_idx] = 0;
                        }
                    }
                }
            }
        }
        
        if (proc_rank == 0) {
            evals_file.close();
            bmat_file.close();
            dmat_file.close();
        }

        MPI_Finalize();
    } catch (std::exception &ex) {
        std::cerr << "\nException : " << ex.what() << "\n\nPlease report this error to the developers through our GitHub repository: https://github.com/sgreene8/FRIES/ \n\n";
    }
    return 0;
}
