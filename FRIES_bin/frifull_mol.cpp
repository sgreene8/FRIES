/*! \file
 *
 * \brief FRI algorithm with systematic vector compression for a molecular
 * system
 */

#include <chrono>
#include <FRIES/io_utils.hpp>
#include <FRIES/compress_utils.hpp>
#include <FRIES/Ext_Libs/argparse.hpp>
#include <FRIES/Hamiltonians/molecule.hpp>
#include <stdexcept>

struct MyArgs : public argparse::Args {
    std::string &hf_path = kwarg("hf_path", "Path to the directory that contains the HF output files eris.txt, hcore.txt, symm.txt, hf_en.txt, and sys_params.txt");
    double &target_norm = kwarg("target", "Target one-norm of solution vector").set_default(0);
    uint32_t &max_iter = kwarg("max_iter", "Maximum number of iterations to run the calculation").set_default(1000000);
    uint32_t &target_nonz = kwarg("vec_nonz", "Target number of nonzero vector elements to keep after each iteration");
    std::string &result_dir = kwarg("result_dir", "Directory in which to save output files").set_default<std::string>("./");
    uint32_t &max_n_dets = kwarg("max_dets", "Maximum number of determinants on a single MPI process");
    std::shared_ptr<std::string> &load_dir = kwarg("load_dir", "Directory from which to load checkpoint files from a previous FRI calculation (in binary format, see documentation for DistVec::save() and DistVec::load())");
    std::shared_ptr<std::string> &ini_path = kwarg("ini_vec", "Prefix for files containing the vector with which to initialize the calculation (files must have names <ini_vec>dets and <ini_vec>vals and be text files)");
    std::shared_ptr<std::string> &trial_path = kwarg("trial_vec", "Prefix for files containing the vector with which to calculate the energy (files must have names <trial_vec>dets and <trial_vec>vals and be text)");
};

int main(int argc, char * argv[]) {
    MyArgs args = argparse::parse<MyArgs>(argc, argv);
    
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
        
        // Rn generator
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::cout << "seed on process " << proc_rank << " is " << seed << std::endl;
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
        double last_one_norm = 0;
        
        if (args.load_dir != nullptr) {
            load_proc_hash(*args.load_dir, proc_scrambler.data());
        }
        else {
            if (proc_rank == 0) {
                for (det_idx = 0; det_idx < 2 * n_orb; det_idx++) {
                    proc_scrambler[det_idx] = mt_obj();
                }
                save_proc_hash(args.result_dir, proc_scrambler.data(), 2 * n_orb);
            }

            MPI_Bcast(proc_scrambler.data(), 2 * n_orb, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        }
        std::vector<uint32_t> vec_scrambler(2 * n_orb);
        for (det_idx = 0; det_idx < 2 * n_orb; det_idx++) {
            vec_scrambler[det_idx] = mt_obj();
        }
        DistVec<double> sol_vec(args.max_n_dets, adder_size, n_orb * 2, n_elec_unf, n_procs, diag_shortcut, 2, proc_scrambler, vec_scrambler);
        
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
        Matrix<uint8_t> &load_dets = sol_vec.indices();
        double *load_vals = (double *)sol_vec.values();
        if (args.trial_path != nullptr) { // load trial vector from file
            n_trial = load_vec_txt(*args.trial_path, load_dets, load_vals);
        }
        else {
            n_trial = 1;
        }
        size_t tot_trial = sum_mpi((int) n_trial, proc_rank, n_procs) / n_procs + 1;
        DistVec<double> trial_vec(tot_trial, tot_trial, n_orb * 2, n_elec_unf, n_procs, proc_scrambler, vec_scrambler);
        if (args.trial_path != nullptr) { // load trial vector from file
            for (det_idx = 0; det_idx < n_trial; det_idx++) {
                if (!trial_vec.add(load_dets[det_idx], load_vals[det_idx], 1)) {
                    throw std::runtime_error("Insufficient memory allocated in adder");
                }
            }
        }
        else { // Otherwise, use HF as trial vector
            if (hf_proc == proc_rank) {
                trial_vec.add(hf_det, 1, 1);
            }
        }
        trial_vec.perform_add(0);
        
        trial_vec.collect_procs();
        std::vector<uintmax_t> trial_hashes(trial_vec.curr_size());
        for (det_idx = 0; det_idx < trial_vec.curr_size(); det_idx++) {
            trial_hashes[det_idx] = sol_vec.idx_to_hash(trial_vec.indices()[det_idx], tmp_orbs);
        }
        
        std::string file_path;
        std::ofstream num_file;
        std::ofstream den_file;
        std::ofstream shift_file;
        std::ofstream norm_file;
        std::ofstream nkept_file;
        std::ofstream prob_file;
        
#pragma mark Initialize solution vector
        if (args.load_dir != nullptr) {
            sol_vec.load(*args.load_dir);
            
            file_path = *args.load_dir;
            file_path.append("S.txt");
            std::ifstream shift_in(file_path);
            if (shift_in.is_open()) {
                // load energy shift (see https://stackoverflow.com/questions/11876290/c-fastest-way-to-read-only-last-line-of-text-file)
                shift_in.seekg(-1, std::ios_base::end);

                bool keepLooping = true;
                while(keepLooping) {
                    char ch;
                    shift_in.get(ch);

                    if((int)shift_in.tellg() <= 1) {
                        shift_in.seekg(0);
                        keepLooping = false;
                    }
                    else if(ch == '\n') {
                        keepLooping = false;
                    }
                    else {
                        shift_in.seekg(-2, std::ios_base::cur);
                    }
                }
                
                shift_in >> en_shift;

                shift_in.close();
            }
        }
        else if (args.ini_path != nullptr) {
            Matrix<uint8_t> load_dets(args.max_n_dets, det_size);
            double *load_vals = (double *)sol_vec.values();
            
            size_t n_dets = load_vec_txt(*args.ini_path, load_dets, load_vals);
            
            for (det_idx = 0; det_idx < n_dets; det_idx++) {
                if (!sol_vec.add(load_dets[det_idx], load_vals[det_idx], 1)) {
                    throw std::runtime_error("Insufficient memory allocated in adder");
                }
            }
            n_dets++; // just to be safe
            bzero(load_vals, n_dets * sizeof(double));
        }
        else {
            if (hf_proc == proc_rank) {
                sol_vec.add(hf_det, 100, 1);
            }
        }
        sol_vec.perform_add(0);
        loc_norm = sol_vec.local_norm();
        glob_norm = sum_mpi(loc_norm, proc_rank, n_procs);
        if (args.load_dir != nullptr) {
            last_one_norm = glob_norm;
        }
        
        if (proc_rank == hf_proc) {
            // Setup output files
            file_path = args.result_dir;
            file_path.append("projnum.txt");
            num_file.open(file_path, std::ofstream::app);
            if (!num_file.is_open()) {
                std::string msg("Could not open file for writing in directory ");
                msg.append(args.result_dir);
                throw std::runtime_error(msg);
            }
            
            file_path = args.result_dir;
            file_path.append("projden.txt");
            den_file.open(file_path, std::ofstream::app);
            
            file_path = args.result_dir;
            file_path.append("S.txt");
            shift_file.open(file_path, std::ofstream::app);
            
            file_path = args.result_dir;
            file_path.append("norm.txt");
            norm_file.open(file_path, std::ofstream::app);
            
            file_path = args.result_dir;
            file_path.append("nkept.txt");
            nkept_file.open(file_path, std::ofstream::app);
            
            file_path = args.result_dir;
            file_path.append("probs.txt");
            nkept_file.open(file_path, std::ofstream::app);
            
            file_path = args.result_dir;
            file_path.append("params.txt");
            std::ofstream param_f(file_path);
            param_f << "FRI calculation\nHF path: " << args.hf_path << "\nepsilon (imaginary time step): " << eps << "\nTarget norm " << target_norm << "\nVector nonzero: " << args.target_nonz << "\n";
            if (args.load_dir != nullptr) {
                param_f << "Restarting calculation from " << args.load_dir << "\n";
            }
            else if (args.ini_path != nullptr) {
                param_f << "Initializing calculation from vector files with prefix " << args.ini_path << '\n';
            }
            else {
                param_f << "Initializing calculation from HF unit vector\n";
            }
            param_f.close();
        }
        
        // Parameters for systematic sampling
        double rn_sys = 0;
        double loc_norms[n_procs];
        size_t max_n_dets = sol_vec.max_size();
        std::vector<size_t> srt_arr(max_n_dets);
        std::vector<bool> keep_exact(max_n_dets, false);
        int vec_idx = 0;
        
        for (unsigned int iterat = 0; iterat < args.max_iter; iterat++) {
            double denom = sol_vec.dot(trial_vec.indices(), trial_vec.values(), trial_vec.curr_size(), trial_hashes);
            denom = sum_mpi(denom, proc_rank, n_procs);
            
#pragma mark Vector compression step
            unsigned int n_samp = args.target_nonz;
            loc_norms[proc_rank] = find_preserve(sol_vec.values(), srt_arr, keep_exact, sol_vec.curr_size(), &n_samp, &glob_norm);
            
            MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, loc_norms, 1, MPI_DOUBLE, MPI_COMM_WORLD);
            if (proc_rank == hf_proc) {
                nkept_file << args.target_nonz - n_samp << '\n';
            }
            
            // Adjust shift
            if ((iterat + 1) % shift_interval == 0) {
                adjust_shift(&en_shift, glob_norm, &last_one_norm, target_norm, shift_damping / shift_interval / eps);
                if (proc_rank == hf_proc) {
                    shift_file << en_shift << "\n";
                    norm_file << glob_norm << "\n";
                }
            }
            
            if (proc_rank == 0) {
                rn_sys = mt_obj() / (1. + UINT32_MAX);
            }
            sys_comp(sol_vec.values(), sol_vec.curr_size(), loc_norms, n_samp, keep_exact, rn_sys);
            for (det_idx = 0; det_idx < sol_vec.curr_size(); det_idx++) {
                if (keep_exact[det_idx]) {
                    sol_vec.del_at_pos(det_idx);
                    keep_exact[det_idx] = 0;
                }
            }
            
            h_op_diag(sol_vec, !vec_idx, 1 + eps * en_shift, -eps);
            sol_vec.set_curr_vec_idx(vec_idx);
            h_op_offdiag(sol_vec, symm, tot_orb, *eris, *h_core, orb_indices, n_frz, n_elec_unf, !vec_idx, -eps);
            vec_idx = !vec_idx;
            
            double numer = sol_vec.dot(trial_vec.indices(), trial_vec.values(), trial_vec.curr_size(), trial_hashes);
            numer = sum_mpi(numer, proc_rank, n_procs);
            numer = ((1 + eps * en_shift) * denom - numer) / eps;
            if (proc_rank == hf_proc) {
                num_file << numer << '\n';
                den_file << denom << "\n";
                std::cout << iterat << ", en est: " << numer / denom << ", shift: " << en_shift << ", norm: " << glob_norm << '\n';
            }
            
            size_t new_max_dets = sol_vec.max_size();
            if (new_max_dets > max_n_dets) {
                keep_exact.resize(new_max_dets, false);
                srt_arr.resize(new_max_dets);
                max_n_dets = new_max_dets;
            }
            
            if ((iterat + 1) % save_interval == 0) {
                sol_vec.save(args.result_dir);
                if (proc_rank == hf_proc) {
                    num_file.flush();
                    den_file.flush();
                    shift_file.flush();
                    nkept_file.flush();
                    prob_file.flush();
                }
            }
        }
        sol_vec.save(args.result_dir);
        if (proc_rank == hf_proc) {
            num_file.close();
            den_file.close();
            shift_file.close();
            nkept_file.close();
            prob_file.close();
        }
            
        MPI_Finalize();
    } catch (std::exception &ex) {
        std::cerr << "\nException : " << ex.what() << "\n\nPlease send a description of this error, a copy of the command-line arguments used, and the random number generator seeds printed for each process to the developers through our GitHub repository: https://github.com/sgreene8/FRIES/ \n\n";
    }
    return 0;
}

