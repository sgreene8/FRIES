/*! \file
 *
 * \brief Perform FRI without matrix compression on the Hubbard-Holstein model, using systematic vector compression.
 *
 */

#include <cstdlib>
#include <chrono>
#define __STDC_LIMIT_MACROS
#include <cstdint>
#include <FRIES/io_utils.hpp>
#include <FRIES/compress_utils.hpp>
#include <FRIES/Ext_Libs/argparse.hpp>
#include <FRIES/Hamiltonians/hub_holstein.hpp>
#include <FRIES/hh_vec.hpp>
#include <stdexcept>

struct MyArgs : public argparse::Args {
    std::string &params_path = kwarg("params_path", "Path to the file that contains the parameters defining the Hamiltonian, number of electrons, number of sites, etc.");
    uint32_t &target_nonz = kwarg("vec_nonz", "Target number of nonzero vector elements to keep after each compression operation");
    uint32_t &max_n_dets = kwarg("max_dets", "Maximum number of determinants on a single MPI process");
    std::shared_ptr<std::string> &load_dir = kwarg("load_dir", "Directory from which to load checkpoint files from a previous FRI calculation (in binary format, see documentation for DistVec::save() and DistVec::load())");
    std::string &result_dir = kwarg("result_dir", "Directory in which to save output files").set_default<std::string>("./");
    double &init_thresh = kwarg("initiator", "Magnitude of vector element required to make it an initiator").set_default(0);
    double &target_norm = kwarg("target", "Target one-norm of solution vector").set_default(0);
    uint32_t &max_iter = kwarg("max_iter", "Maximum number of iterations to run the calculation").set_default(1000000);
};

int main(int argc, char * argv[]) {
    MyArgs args = argparse::parse<MyArgs>(argc, argv);
    
    try {
        int n_procs = 1;
        int proc_rank = 0;
        unsigned int proc_idx, ref_proc;
        MPI_Init(NULL, NULL);
        MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
        
        double target_norm = args.target_norm;
        size_t max_n_dets = args.max_n_dets;
        
        // Parameters
        double shift_damping = 0.05;
        unsigned int shift_interval = 10;
        unsigned int save_interval = 1000;
        double en_shift = 0;
        
        // Read in data files
        hh_input in_data;
        parse_hh_input(args.params_path, &in_data);
        double eps = in_data.eps;
        unsigned int hub_len = in_data.lat_len;
        unsigned int hub_dim = in_data.n_dim;
        unsigned int n_elec = in_data.n_elec;
        double hub_t = 1;
        double hub_u = in_data.elec_int;
        double ph_freq = in_data.ph_freq;
        double elec_ph = in_data.elec_ph;
        double hf_en = in_data.hf_en;
        
        if (hub_dim != 1) {
            fprintf(stderr, "Error: only 1-D Hubbard calculations supported right now.\n");
            return 0;
        }
        
        unsigned int n_orb = hub_len;
        
        // Rn generator
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::cout << "seed on process " << proc_rank << " is " << seed << std::endl;
        std::mt19937 mt_obj((unsigned int)seed);
        
        // Initialize hash function for processors and vector
        std::vector<uint32_t> proc_scrambler(2 * n_orb);
        double loc_norm, glob_norm;
        double last_one_norm = 0;
        
        if (args.load_dir != nullptr) {
            load_proc_hash(*args.load_dir, proc_scrambler.data());
        }
        else {
            if (proc_rank == 0) {
                for (size_t orb_idx = 0; orb_idx < 2 * n_orb; orb_idx++) {
                    proc_scrambler[orb_idx] = mt_obj();
                }
                save_proc_hash(args.result_dir, proc_scrambler.data(), 2 * n_orb);
            }
            MPI_Bcast(proc_scrambler.data(), 2 * n_orb, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        }
        std::vector<uint32_t> vec_scrambler(2 * n_orb);
        for (size_t orb_idx = 0; orb_idx < 2 * n_orb; orb_idx++) {
            vec_scrambler[orb_idx] = mt_obj();
        }
        
        // Solution vector
        size_t spawn_length = n_elec * 4 * max_n_dets / n_procs;
        if (spawn_length > 200000) {
            spawn_length = 200000;
        }
        uint8_t ph_bits = 3;
        std::function<double(const uint8_t *)> diag_shortcut = [hub_len](const uint8_t *det) {
            return hub_diag((uint8_t *)det, hub_len);
        };
        HubHolVec<double> sol_vec(max_n_dets, spawn_length, hub_len, ph_bits, n_elec, n_procs, diag_shortcut, 2, proc_scrambler, vec_scrambler);
        size_t det_size = CEILING(2 * n_orb + ph_bits * n_orb, 8);
        
        uint8_t neel_det[det_size];
        gen_neel_det_1D(n_orb, n_elec, ph_bits, neel_det);
        ref_proc = sol_vec.idx_to_proc(neel_det);
        uint8_t neel_occ[n_elec];
        sol_vec.gen_orb_list(neel_det, neel_occ);
        
        // Initialize solution vector
        if (args.load_dir != nullptr) {
            sol_vec.load(*args.load_dir);
        }
        else {
            if (ref_proc == proc_rank) {
                sol_vec.add(neel_det, 100, 1);
            }
        }
        sol_vec.perform_add(0);
        loc_norm = sol_vec.local_norm();
        glob_norm = sum_mpi(loc_norm, proc_rank, n_procs);
        if (args.load_dir != nullptr) {
            last_one_norm = glob_norm;
        }
        
        std::string file_path;
        std::ofstream norm_file;
        std::ofstream num_file;
        std::ofstream den_file;
        std::ofstream shift_file;
        if (proc_rank == ref_proc) {
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
            file_path.append("params.txt");
            std::ofstream param_f(file_path);
            param_f << "FRI calculation\nHubbard-Holstein parameters path: " << args.params_path << "\nepsilon (imaginary time step): " << eps << "\nTarget norm " << target_norm << "\nInitiator threshold: " << args.init_thresh << "\nVector nonzero: " << args.target_nonz << "\n";
            if (args.load_dir != nullptr) {
                param_f << "Restarting calculation from " << args.load_dir << "\n";
            }
            else {
                param_f << "Initializing calculation from Neel unit vector\n";
            }
            param_f.close();
        }
        
        // Parameters for systematic sampling
        double loc_norms[n_procs];
        double rn_sys = 0;
        std::vector<size_t> srt_arr(max_n_dets);
        std::vector<bool> keep_exact(max_n_dets, false);
        
        int ini_flag;
        double matr_el;
        double recv_nums[n_procs];
        uint8_t new_det[det_size];
        
        uint8_t (*spawn_orbs)[2] = (uint8_t (*)[2])malloc(sizeof(uint8_t) * n_elec * 2 * 2);
        
        unsigned int iterat;
        Matrix<uint8_t> &neighb_orbs = sol_vec.neighb();
        for (iterat = 0; iterat < args.max_iter; iterat++) {
            size_t det_idx = 0;
            int num_added = 1;
            size_t vec_size = sol_vec.curr_size();
            size_t adder_size = sol_vec.adder_size() - n_elec * 4;
            
            double *vals_before_mult = sol_vec.values();
            sol_vec.set_curr_vec_idx(1);
            sol_vec.zero_vec();
            
            while (num_added > 0) {
                num_added = 0;
                while (det_idx < vec_size && num_added < adder_size) {
                    double curr_el = vals_before_mult[det_idx];
                    if (curr_el == 0) {
                        det_idx++;
                        continue;
                    }
                    uint8_t *curr_det = sol_vec.indices()[det_idx];
                    ini_flag = fabs(curr_el) > args.init_thresh;
                    
                    size_t n_success = hub_all(n_elec, neighb_orbs[det_idx], spawn_orbs);
                    
                    for (size_t ex_idx = 0; ex_idx < n_success; ex_idx++) {
                        memcpy(new_det, curr_det, det_size);
                        zero_bit(new_det, spawn_orbs[ex_idx][0]);
                        set_bit(new_det, spawn_orbs[ex_idx][1]);
                        sol_vec.add(new_det, eps * hub_t * curr_el, ini_flag);
                    }
                    num_added += n_success;
                    
                    uint8_t *curr_occ = sol_vec.orbs_at_pos(det_idx);
                    uint8_t *curr_phonons = sol_vec.phonons_at_pos(det_idx);
                    for (size_t elec_idx = 0; elec_idx < n_elec / 2; elec_idx++) {
                        uint8_t site = curr_occ[elec_idx];
                        uint8_t phonon_num = curr_phonons[site];
                        int doubly_occ = read_bit(curr_det, site + hub_len);
                        if (phonon_num > 0) {
                            sol_vec.det_from_ph(curr_det, new_det, site, -1);
                            sol_vec.add(new_det, -eps * elec_ph * sqrt(phonon_num) * (doubly_occ + 1) * curr_el, ini_flag);
                            num_added++;
                        }
                        if (phonon_num + 1 < (1 << ph_bits)) {
                            sol_vec.det_from_ph(curr_det, new_det, site, +1);
                            sol_vec.add(new_det, -eps * elec_ph * sqrt(phonon_num + 1) * (doubly_occ + 1) * curr_el, ini_flag);
                            num_added++;
                        }
                    }
                    for (size_t elec_idx = n_elec / 2; elec_idx < n_elec; elec_idx++) {
                        uint8_t site = curr_occ[elec_idx] - n_orb;
                        int doubly_occ = read_bit(curr_det, site);
                        if (!doubly_occ) {
                            uint8_t phonon_num = curr_phonons[site];
                            if (phonon_num > 0) {
                                sol_vec.det_from_ph(curr_det, new_det, site, -1);
                                sol_vec.add(new_det, -eps * elec_ph * sqrt(phonon_num) * curr_el, ini_flag);
                                num_added++;
                            }
                            if (phonon_num + 1 < (1 << ph_bits)) {
                                sol_vec.det_from_ph(curr_det, new_det, site, +1);
                                sol_vec.add(new_det, -eps * elec_ph * sqrt(phonon_num + 1) * curr_el, ini_flag);
                                num_added++;
                            }
                        }
                    }
                    det_idx++;
                }
                num_added = sum_mpi(num_added, proc_rank, n_procs);
                sol_vec.perform_add(0);
                sol_vec.set_curr_vec_idx(0);
                vals_before_mult = sol_vec.values();
                sol_vec.set_curr_vec_idx(1);
            }
            sol_vec.set_curr_vec_idx(0);
            for (size_t det_idx = 0; det_idx < vec_size; det_idx++) {
                double *curr_el = sol_vec[det_idx];
                // Death/cloning step
                if (*curr_el != 0) {
                    double diag_el = sol_vec.matr_el_at_pos(det_idx);
                    double phonon_diag = sol_vec.total_ph(det_idx) * ph_freq;
                    *curr_el *= 1 - eps * (diag_el * hub_u + phonon_diag - hf_en - en_shift);
                }
            }
            sol_vec.add_vecs(0, 1);
            
            size_t new_max_dets = sol_vec.max_size();
            if (new_max_dets > max_n_dets) {
                keep_exact.resize(new_max_dets, false);
                srt_arr.resize(new_max_dets);
                max_n_dets = new_max_dets;
            }
            
            // Compression step
            unsigned int n_samp = args.target_nonz;
            loc_norms[proc_rank] = find_preserve(sol_vec.values(), srt_arr, keep_exact, sol_vec.curr_size(), &n_samp, &glob_norm);
            
            // Adjust shift
            if ((iterat + 1) % shift_interval == 0) {
                adjust_shift(&en_shift, glob_norm, &last_one_norm, target_norm, shift_damping / shift_interval / eps);
                if (proc_rank == ref_proc) {
                    shift_file << en_shift << '\n';
                    norm_file << glob_norm << '\n';
                }
            }
            
            // Calculate energy estimate
            matr_el = calc_ref_ovlp(sol_vec.indices(), sol_vec.values(), sol_vec.phonon_nums(), sol_vec.curr_size(), neel_det, neel_occ, n_elec, hub_len, elec_ph / hub_t);
            MPI_Gather(&matr_el, 1, MPI_DOUBLE, recv_nums, 1, MPI_DOUBLE, ref_proc, MPI_COMM_WORLD);
            if (proc_rank == ref_proc) {
                double diag_el = sol_vec.matr_el_at_pos(0);
                double ref_element = *(sol_vec[0]);
                matr_el = (diag_el * hub_u - hf_en) * ref_element;
                for (proc_idx = 0; proc_idx < n_procs; proc_idx++) {
                    matr_el += recv_nums[proc_idx] * -hub_t;
                }
                num_file << matr_el << '\n';
                den_file << ref_element << '\n';
                std::cout << iterat << ", norm: " << glob_norm << ", en est: " << matr_el / ref_element << ", shift: " << en_shift << ", n_neel: " << ref_element << '\n';
            }
            
            if (proc_rank == 0) {
                rn_sys = mt_obj() / (1. + UINT32_MAX);
            }
            MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, loc_norms, 1, MPI_DOUBLE, MPI_COMM_WORLD);
            sys_comp(sol_vec.values(), sol_vec.curr_size(), loc_norms, n_samp, keep_exact, rn_sys);
            for (size_t det_idx = 0; det_idx < sol_vec.curr_size(); det_idx++) {
                if (keep_exact[det_idx] && !(proc_rank == 0 && det_idx == 0)) {
                    sol_vec.del_at_pos(det_idx);
                    keep_exact[det_idx] = 0;
                }
            }
            
            // Save vector snapshot to disk
            if ((iterat + 1) % save_interval == 0) {
                sol_vec.save(args.result_dir);
                if (proc_rank == ref_proc) {
                    num_file.flush();
                    den_file.flush();
                    shift_file.flush();
                }
            }
        }
        sol_vec.save(args.result_dir);
        if (proc_rank == ref_proc) {
            num_file.close();
            den_file.close();
            shift_file.close();
        }
        MPI_Finalize();
    } catch (std::exception &ex) {
        std::cerr << "\nException : " << ex.what() << "\n\nPlease send a description of this error, a copy of the command-line arguments used, and the random number generator seeds printed for each process to the developers through our GitHub repository: https://github.com/sgreene8/FRIES/ \n\n";
    }
    return 0;
}

