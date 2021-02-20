/*! \file
 *
 * \brief FRI algorithm with multinomial matrix compression for a molecular
 * system
 */

#include <FRIES/Hamiltonians/near_uniform.hpp>
#include <FRIES/io_utils.hpp>
#include <chrono>
#include <FRIES/compress_utils.hpp>
#include <FRIES/Ext_Libs/argparse.hpp>
#include <FRIES/Hamiltonians/heat_bathPP.hpp>
#include <FRIES/Hamiltonians/molecule.hpp>
#include <stdexcept>

struct MyArgs : public argparse::Args {
    std::string hf_path = kwarg("hf_path", "Path to the directory that contains the HF output files eris.txt, hcore.txt, symm.txt, hf_en.txt, and sys_params.txt");
    double target_norm = kwarg("target", "Target one-norm of solution vector").set_default(0);
    std::string dist_str = kwarg("distribution", "Hamiltonian factorization to use, either near-uniform (NU) or heat-bath Power-Pitzer (HB)");
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
    
    h_dist fri_dist;
    try {
        if (args.dist_str == "HB") {
            fri_dist = heat_bath;
        }
        else if (args.dist_str == "HB") {
            fri_dist = heat_bath;
        }
        else {
            throw std::runtime_error("\"dist_str\" argument must be either \"NU\" or \"HB\"");
        }
    } catch (std::exception &ex) {
        std::cerr << "\nError parsing command line: " << ex.what() << "\n\n";
        return 1;
    }
    
    double target_norm = args.target_norm;
    
    int n_procs = 1;
    int proc_rank = 0;
    unsigned int proc_idx, hf_proc;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    try {
        // Parameters
        double shift_damping = 0.05;
        unsigned int shift_interval = 10;
        unsigned int save_interval = 1000;
        double en_shift = 0;
        
        // Read in data files
        hf_input in_data;
        parse_hf_input(args.hf_path, &in_data);
        double eps = in_data.eps;
        unsigned int n_elec = in_data.n_elec;
        unsigned int n_frz = in_data.n_frz;
        unsigned int n_orb = in_data.n_orb;
        double hf_en = in_data.hf_en;;
        
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
        unsigned int spawn_length = args.matr_samp * 2 / n_procs / n_procs;
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
        double loc_norms[n_procs];
        double glob_norm;
        
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
        DistVec<double> sol_vec(args.max_n_dets, spawn_length, n_orb * 2, n_elec_unf, n_procs, diag_shortcut, NULL, 1, proc_scrambler, vec_scrambler);
        
        uint8_t hf_det[det_size];
        gen_hf_bitstring(n_orb, n_elec - n_frz, hf_det);
        hf_proc = sol_vec.idx_to_proc(hf_det);
        
        uint8_t tmp_orbs[n_elec_unf];
        unsigned int max_spawn = args.matr_samp; // should scale as max expected # from one determinant
        uint8_t *spawn_orbs = (uint8_t *)malloc(sizeof(uint8_t) * 4 * max_spawn);
        double *spawn_probs = (double *) malloc(sizeof(double) * max_spawn);
        uint8_t (*sing_orbs)[2] = (uint8_t (*)[2]) spawn_orbs;
        uint8_t (*doub_orbs)[4] = (uint8_t (*)[4]) spawn_orbs;
        
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
            trial_vec.add(hf_det, 1, 1);
            htrial_vec.add(hf_det, 1, 1);
        }
        trial_vec.perform_add();
        htrial_vec.perform_add();
        
        trial_vec.collect_procs();
        std::vector<uintmax_t> trial_hashes(trial_vec.curr_size());
        for (det_idx = 0; det_idx < trial_vec.curr_size(); det_idx++) {
            trial_hashes[det_idx] = sol_vec.idx_to_hash(trial_vec.indices()[det_idx], tmp_orbs);
        }
        
        // Calculate H * trial vector, and accumulate results on each processor
        h_op_offdiag(htrial_vec, symm, tot_orb, *eris, *h_core, spawn_orbs, n_frz, n_elec_unf, 1, 1);
        htrial_vec.set_curr_vec_idx(0);
        h_op_diag(htrial_vec, 0, 0, 1);
        htrial_vec.add_vecs(0, 1);
        htrial_vec.collect_procs();
        std::vector<uintmax_t> htrial_hashes(htrial_vec.curr_size());
        for (det_idx = 0; det_idx < htrial_vec.curr_size(); det_idx++) {
            htrial_hashes[det_idx] = sol_vec.idx_to_hash(htrial_vec.indices()[det_idx], tmp_orbs);
        }
        
        // Count # single/double excitations from HF
        sol_vec.gen_orb_list(hf_det, tmp_orbs);
        size_t n_hf_doub = doub_ex_symm(hf_det, tmp_orbs, n_elec_unf, n_orb, doub_orbs, symm);
        size_t n_hf_sing = count_singex(hf_det, tmp_orbs, symm, n_orb, symm_lookup, n_elec_unf);
        double p_doub = (double) n_hf_doub / (n_hf_sing + n_hf_doub);
        
        std::string file_path;
        std::ofstream norm_file;
        std::ofstream num_file;
        std::ofstream den_file;
        std::ofstream shift_file;
        std::ofstream ini_file;
        
        // Initialize solution vector
        if (args.load_dir != nullptr) {
            sol_vec.load(*args.load_dir);
            
            file_path = *args.load_dir;
            file_path.append("S.txt");
            std::ifstream shift_in(file_path);
            if(shift_in.is_open()) {

                // load energy shift (seehttps://stackoverflow.com/questions/11876290/c-fastest-way-to-read-only-last-line-of-text-file)
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
            // from an initial vector in .txt format
            Matrix<uint8_t> &load_dets = sol_vec.indices();
            double *load_vals = sol_vec.values();
            
            size_t n_dets = load_vec_txt(*args.ini_path, load_dets, load_vals);
            
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
        loc_norms[proc_rank] = sol_vec.local_norm();

        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, loc_norms, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        glob_norm = 0;
        for (proc_idx = 0; proc_idx < n_procs; proc_idx++) {
            glob_norm += loc_norms[proc_idx];
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
            file_path.append("nini.txt");
            ini_file.open(file_path, std::ofstream::app);
            
            file_path = args.result_dir;
            file_path.append("params.txt");
            std::ofstream param_f(file_path);
            param_f << "FRI calculation\nHF path: " << args.hf_path << "\nepsilon (imaginary time step): " << eps << "\nTarget norm " << target_norm << "\nInitiator threshold: " << args.init_thresh << "\nMatrix nonzero: " << args.matr_samp << "\nVector nonzero: " << args.target_nonz << "\n";
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
        
        hb_info *hb_probs = NULL;
        if (fri_dist == heat_bath) {
            hb_probs = set_up(tot_orb, n_orb, *eris);
        }
        
        int ini_flag;
        unsigned int n_walk, n_doub, n_sing;
        double last_one_norm = 0;
        uint8_t new_det[det_size];
        double matr_el;
        double recv_nums[n_procs];
        double recv_dens[n_procs];
        
        // Variables for compression
        double rn_sys = 0;
        double lbound;
        double weight;
        std::vector<size_t> srt_arr(args.max_n_dets);
        for (det_idx = 0; det_idx < args.max_n_dets; det_idx++) {
            srt_arr[det_idx] = det_idx;
        }
        std::vector<bool> keep_exact(args.max_n_dets, false);
        
        unsigned int iterat;
        size_t n_ini;
        for (iterat = 0; iterat < args.max_iter; iterat++) {
            n_ini = 0;
            
            // Systematic sampling to determine number of samples for each column
            if (proc_rank == 0) {
                rn_sys = mt_obj() / (1. + UINT32_MAX);
            }

            MPI_Bcast(&rn_sys, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            unsigned int curr_mat_samp = (iterat < 10) ? args.matr_samp / 10 : args.matr_samp;
            lbound = seed_sys(loc_norms, &rn_sys, curr_mat_samp);
            for (det_idx = 0; det_idx < sol_vec.curr_size(); det_idx++) {
                double *curr_el = sol_vec[det_idx];
                uint8_t *curr_det = sol_vec.indices()[det_idx];
                weight = fabs(*curr_el);
                if (weight == 0) {
                    continue;
                }
                n_walk = 0;
                lbound += weight;
                while (rn_sys < lbound) {
                    n_walk++;
                    rn_sys += glob_norm / (curr_mat_samp);
                }
                double colsamp_wt = weight / (glob_norm / curr_mat_samp);
                if (colsamp_wt > 1) {
                    colsamp_wt = 1;
                }
                
                ini_flag = weight > args.init_thresh;
                n_ini += ini_flag;
                
                // spawning step
                uint8_t *occ_orbs = sol_vec.orbs_at_pos(det_idx);
                count_symm_virt(unocc_symm_cts, occ_orbs, n_elec_unf,
                                n_orb, n_irreps, symm_lookup, symm);
                n_doub = bin_sample(n_walk, p_doub, mt_obj);
                n_sing = n_walk - n_doub;
                
                if (n_doub > max_spawn || n_sing / 2 > max_spawn) {
                    printf("Allocating more memory for spawning\n");
                    max_spawn *= 2;
                    spawn_orbs = (uint8_t *)realloc(spawn_orbs, sizeof(uint8_t) * 4 * max_spawn);
                    spawn_probs = (double *)realloc(spawn_probs, sizeof(double) * max_spawn);
                }
                
                if (fri_dist == near_uni) {
                    n_doub = doub_multin(curr_det, occ_orbs, n_elec_unf, symm, n_orb, symm_lookup, unocc_symm_cts, n_doub, mt_obj, doub_orbs, spawn_probs);
                }
                else if (fri_dist == heat_bath) {
                    n_doub = hb_doub_multi(curr_det, occ_orbs, n_elec_unf, symm, hb_probs, symm_lookup, n_doub, mt_obj, doub_orbs, spawn_probs);
                }
                
                size_t walker_idx;
                for (walker_idx = 0; walker_idx < n_doub; walker_idx++) {
                    matr_el = doub_matr_el_nosgn(doub_orbs[walker_idx], tot_orb, *eris, n_frz);
                    if (fabs(matr_el) > 1e-9) {
                        memcpy(new_det, curr_det, det_size);
                        matr_el *= -eps / spawn_probs[walker_idx] / p_doub / n_walk * (*curr_el) * doub_det_parity(new_det, doub_orbs[walker_idx]) / colsamp_wt;
                        sol_vec.add(new_det, matr_el, ini_flag);
                    }
                }
                
                n_sing = sing_multin(curr_det, occ_orbs, n_elec_unf, symm, n_orb, symm_lookup, unocc_symm_cts, n_sing, mt_obj, sing_orbs, spawn_probs);
                
                for (walker_idx = 0; walker_idx < n_sing; walker_idx++) {
                    matr_el = sing_matr_el_nosgn(sing_orbs[walker_idx], occ_orbs, tot_orb, *eris, *h_core, n_frz, n_elec_unf);
                    if (fabs(matr_el) > 1e-9) {
                        memcpy(new_det, curr_det, det_size);
                        matr_el *= -eps / spawn_probs[walker_idx] / (1 - p_doub) / n_walk * (*curr_el) * sing_det_parity(new_det, sing_orbs[walker_idx]) / colsamp_wt;
                        sol_vec.add(new_det, matr_el, ini_flag);
                    }
                }
                
                // Death/cloning step
                double diag_el = sol_vec.matr_el_at_pos(det_idx);
                *curr_el *= 1 - eps * (diag_el - en_shift);
            }
            sol_vec.perform_add();
            
            // Compression step
            unsigned int n_samp = args.target_nonz;
            loc_norms[proc_rank] = find_preserve(sol_vec.values(), srt_arr, keep_exact, sol_vec.curr_size(), &n_samp, &glob_norm);
            
            // Adjust shift
            if ((iterat + 1) % shift_interval == 0) {
                adjust_shift(&en_shift, glob_norm, &last_one_norm, target_norm, shift_damping / shift_interval / eps);
                if (proc_rank == hf_proc) {
                    shift_file << en_shift << "\n";
                    norm_file << glob_norm << "\n";
                }
            }
            matr_el = sol_vec.dot(htrial_vec.indices(), htrial_vec.values(), htrial_vec.curr_size(), htrial_hashes);
            double denom = sol_vec.dot(trial_vec.indices(), trial_vec.values(), trial_vec.curr_size(), trial_hashes);

            MPI_Gather(&matr_el, 1, MPI_DOUBLE, recv_nums, 1, MPI_DOUBLE, hf_proc, MPI_COMM_WORLD);
            MPI_Gather(&denom, 1, MPI_DOUBLE, recv_dens, 1, MPI_DOUBLE, hf_proc, MPI_COMM_WORLD);
            if (proc_rank == hf_proc) {
                matr_el = 0;
                denom = 0;
                for (proc_idx = 0; proc_idx < n_procs; proc_idx++) {
                    matr_el += recv_nums[proc_idx];
                    denom += recv_dens[proc_idx];
                }
                num_file << matr_el << '\n';
                den_file << denom << '\n';
                std::cout << iterat << ", en est: " << matr_el / denom << ", shift: " << en_shift << ", norm: " << glob_norm << '\n';
                ini_file << n_ini << '\n';
            }
            
            if (proc_rank == 0) {
                rn_sys = mt_obj() / (1. + UINT32_MAX);
            }

            MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, loc_norms, 1, MPI_DOUBLE, MPI_COMM_WORLD);
            sys_comp(sol_vec.values(), sol_vec.curr_size(), loc_norms, n_samp, keep_exact, rn_sys);
            for (det_idx = 0; det_idx < sol_vec.curr_size(); det_idx++) {
                if (keep_exact[det_idx] && sol_vec.indices()[det_idx] != hf_det) {
                    sol_vec.del_at_pos(det_idx);
                    keep_exact[det_idx] = 0;
                }
            }
            
            if ((iterat + 1) % save_interval == 0) {
                sol_vec.save(args.result_dir);
                if (proc_rank == hf_proc) {
                    num_file.flush();
                    den_file.flush();
                    shift_file.flush();
                }
            }
        }
        sol_vec.save(args.result_dir);
        if (proc_rank == hf_proc) {
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
