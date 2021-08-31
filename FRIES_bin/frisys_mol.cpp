/*! \file
 *
 * \brief FRI algorithm with systematic matrix compression for a molecular
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
    std::string &fcidump_path = kwarg("fcidump_path", "Path to FCIDUMP file that contains the integrals defining the Hamiltonian.");
    double &target_norm = kwarg("target", "Target one-norm of solution vector").set_default(0);
    std::string &dist_str = kwarg("distribution", "Hamiltonian factorization to use, either heat-bath Power-Pitzer (HB) or unnormalized heat-bath Power-Pitzer (HB_unnorm)");
    uint32_t &max_iter = kwarg("max_iter", "Maximum number of iterations to run the calculation").set_default(1000000);
    uint32_t &target_nonz = kwarg("vec_nonz", "Target number of nonzero vector elements to keep after each iteration");
    uint32_t &matr_samp = kwarg("mat_nonz", "Target number of nonzero matrix elements to keep after each multiplication by a Hamiltonian matrix factor");
    std::string &result_dir = kwarg("result_dir", "Directory in which to save output files").set_default<std::string>("./");
    uint32_t &max_n_dets = kwarg("max_dets", "Maximum number of determinants on a single MPI process");
    double &init_thresh = kwarg("initiator", "Magnitude of vector element required to make it an initiator").set_default(0);
    std::shared_ptr<std::string> &load_dir = kwarg("load_dir", "Directory from which to load checkpoint files from a previous FRI calculation (in binary format, see documentation for DistVec::save() and DistVec::load())");
    std::shared_ptr<std::string> &ini_path = kwarg("ini_vec", "Prefix for files containing the vector with which to initialize the calculation (files must have names <ini_vec>dets and <ini_vec>vals and be text files)");
    std::shared_ptr<std::string> &trial_path = kwarg("trial_vec", "Prefix for files containing the vector with which to calculate the energy (files must have names <trial_vec>dets and <trial_vec>vals and be text)");
    std::shared_ptr<std::string> &determ_path = kwarg("det_space", "Path to a .txt file containing the determinants used to define the deterministic space to use in a semistochastic calculation.");
    double &epsilon = kwarg("epsilon", "The imaginary time step (\eps) to use when evolving the walker distribution.");
    std::string &point_group = kwarg("point_group", "Specifies the point-group symmetry to assume when reading irrep labels from the FCIDUMP file.").set_default<std::string>("C1");
    std::shared_ptr<double> &ham_shift = kwarg("ham_shift", "The energy by which the diagonal elements of the Hamiltonian are shifted");
};

int main(int argc, char * argv[]) {
    MyArgs args = argparse::parse<MyArgs>(argc, argv);
    
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

        MPI_Init(NULL, NULL);
        MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
        
        size_t max_n_dets = args.max_n_dets;
        uint32_t matr_samp = args.matr_samp;
        
        // Parameters
        double shift_damping = 0.05;
        unsigned int shift_interval = 10;
        unsigned int save_interval = 100;
        double en_shift = 0;
        
        // Read in data files
        fcidump_input *in_data = parse_fcidump(args.fcidump_path, args.point_group);
        double eps = args.epsilon;
        unsigned int n_elec = in_data->n_elec;
        unsigned int n_frz = 0;
        unsigned int n_orb = in_data->n_orb_;
        size_t det_size = CEILING(2 * n_orb, 8);
        
        unsigned int n_elec_unf = n_elec - n_frz;
        unsigned int tot_orb = n_orb + n_frz / 2;
        
        uint8_t *symm = in_data->symm;
        Matrix<double> *h_core = in_data->hcore;
        SymmERIs *eris = &(in_data->eris);

        uint8_t tmp_orbs[n_elec_unf];
        uint8_t hf_det[det_size];
        gen_hf_bitstring(n_orb, n_elec - n_frz, hf_det);
        find_bits(hf_det, tmp_orbs, det_size);
        double hf_en;
        if (args.ham_shift != nullptr) {
            hf_en = *args.ham_shift;
            hf_en -= in_data->core_en;
        }
        else {
            hf_en = diag_matrel(tmp_orbs, tot_orb, *eris, *h_core, n_frz, n_elec);
        }
        
        // Rn generator
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::cout << "seed on process " << proc_rank << " is " << seed << std::endl;
        std::mt19937 mt_obj((unsigned int)seed);
        
        // Solution vector
        unsigned int spawn_length = matr_samp * 4 / n_procs;
        size_t adder_size = spawn_length > 1000000 ? 1000000 : spawn_length;
        std::function<double(const uint8_t *)> diag_shortcut = [tot_orb, eris, h_core, n_frz, n_elec, hf_en](const uint8_t *occ_orbs) {
            return diag_matrel(occ_orbs, tot_orb, *eris, *h_core, n_frz, n_elec) - hf_en;
        };
        std::function<double(uint8_t *, uint8_t *)> sing_shortcut = [tot_orb, eris, h_core, n_frz, n_elec_unf](uint8_t *ex_orbs, uint8_t *occ_orbs) {
            return sing_matr_el_nosgn(ex_orbs, occ_orbs, tot_orb, *eris, *h_core, n_frz, n_elec_unf);
        };
        std::function<double(uint8_t *)> doub_shortcut = [tot_orb, eris, n_frz](uint8_t *ex_orbs) {
            return doub_matr_el_nosgn(ex_orbs, tot_orb, *eris, n_frz);
        };
        
        SymmInfo basis_symm(in_data->symm, n_orb);
        
        // Initialize hash function for processors and vector
        std::vector<uint32_t> proc_scrambler(2 * n_orb);
        double loc_norm, glob_norm;
        double last_norm = 0;
        
        if (args.load_dir != nullptr) {
            load_proc_hash(*args.load_dir, proc_scrambler.data());
        }
        else {
            if (proc_rank == 0) {
                for (size_t det_idx = 0; det_idx < 2 * n_orb; det_idx++) {
                    proc_scrambler[det_idx] = mt_obj();
                }
                save_proc_hash(args.result_dir, proc_scrambler.data(), 2 * n_orb);
            }

            MPI_Bcast(proc_scrambler.data(), 2 * n_orb, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        }
        std::vector<uint32_t> vec_scrambler(2 * n_orb);
        for (size_t det_idx = 0; det_idx < 2 * n_orb; det_idx++) {
            vec_scrambler[det_idx] = mt_obj();
        }
        
        DistVec<double> sol_vec(max_n_dets, adder_size, n_orb * 2, n_elec_unf, n_procs, diag_shortcut, 2, proc_scrambler, vec_scrambler);
        
        size_t n_states = n_elec_unf > (n_orb - n_elec_unf / 2) ? n_elec_unf : n_orb - n_elec_unf / 2;
        HBCompressSys comp_vecs(spawn_length, n_states);
        
        gen_hf_bitstring(n_orb, n_elec - n_frz, hf_det);
        hf_proc = sol_vec.idx_to_proc(hf_det);
        
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
        unsigned int tot_trial = sum_mpi((int) n_trial, proc_rank, n_procs);
        tot_trial = CEILING(tot_trial * 2, n_procs);
        DistVec<double> trial_vec(tot_trial, tot_trial, n_orb * 2, n_elec_unf, n_procs, proc_scrambler, vec_scrambler);
        DistVec<double> htrial_vec(tot_trial * n_ex / n_procs, tot_trial * n_ex / n_procs, n_orb * 2, n_elec_unf, n_procs, diag_shortcut, 2, proc_scrambler, vec_scrambler);
        if (args.trial_path != nullptr) { // load trial vector from file
            for (size_t det_idx = 0; det_idx < n_trial; det_idx++) {
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
        trial_vec.perform_add(0);
        htrial_vec.perform_add(0);
        
        trial_vec.collect_procs();
        std::vector<uintmax_t> trial_hashes(trial_vec.curr_size());
        for (size_t det_idx = 0; det_idx < trial_vec.curr_size(); det_idx++) {
            trial_hashes[det_idx] = sol_vec.idx_to_hash(trial_vec.indices()[det_idx], tmp_orbs);
        }
        
        uint8_t *scratch_orbs;
        size_t scratch_size;
        if (spawn_length < n_ex) {
            scratch_orbs = (uint8_t *)malloc(sizeof(uint8_t) * n_ex * 4);
            scratch_size = 4 * n_ex;
        }
        else {
            scratch_orbs = (uint8_t *)comp_vecs.orb_indices1;
            scratch_size = 4 * spawn_length;
        }
        
        // Calculate H * trial vector, and accumulate results on each processor
        h_op_offdiag(htrial_vec, symm, tot_orb, *eris, *h_core, scratch_orbs, scratch_size, n_frz, n_elec_unf, 1, 1, 0);
        htrial_vec.set_curr_vec_idx(0);
        h_op_diag(htrial_vec, 0, 0, 1);
        htrial_vec.add_vecs(0, 1);
        htrial_vec.collect_procs();
        std::vector<uintmax_t> htrial_hashes(htrial_vec.curr_size());
        for (size_t det_idx = 0; det_idx < htrial_vec.curr_size(); det_idx++) {
            htrial_hashes[det_idx] = sol_vec.idx_to_hash(htrial_vec.indices()[det_idx], tmp_orbs);
        }
        
        // Count # single/double excitations from HF
        sol_vec.gen_orb_list(hf_det, tmp_orbs);
        size_t n_hf_doub = doub_ex_symm(hf_det, tmp_orbs, n_elec_unf, n_orb, (uint8_t (*)[4])scratch_orbs, symm);
        size_t n_hf_sing = count_singex(hf_det, tmp_orbs, n_elec_unf, &basis_symm);
        double p_doub = (double) n_hf_doub / (n_hf_sing + n_hf_doub);
        if (spawn_length < n_ex) {
            free(scratch_orbs);
        }
        
        std::string file_path;
        std::ofstream norm_file;
        std::ofstream num_file;
        std::ofstream den_file;
        std::ofstream shift_file;
        std::ofstream nkept_file;
        std::ofstream ini_file;
        
        size_t n_determ = 0; // Number of deterministic determinants on this process
        if (args.load_dir == nullptr && args.determ_path != nullptr) {
            n_determ = sol_vec.init_dense(*args.determ_path, args.result_dir);
        }
        int dense_sizes[n_procs];
        int determ_tmp = (int) n_determ;

        MPI_Gather(&determ_tmp, 1, MPI_INT, dense_sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (proc_rank == 0 && args.load_dir == nullptr) {
            file_path = args.result_dir;
            file_path.append("dense.txt");
            std::ofstream dense_f(file_path);
            
            if (!dense_f.is_open()) {
                throw std::runtime_error("Error opening file containing sizes of deterministic subspaces");
            }
            for (int proc_idx = 0; proc_idx < n_procs; proc_idx++) {
                dense_f << dense_sizes[proc_idx] << ", ";
            }
            dense_f << '\n';
            dense_f.close();
        }
        
#pragma mark Initialize solution vector
        if (args.load_dir != nullptr) {
            n_determ = sol_vec.load(*args.load_dir);
            
            file_path = *args.load_dir;
            file_path.append("S.txt");
            load_last_line(file_path, &en_shift);
        }
        else if (args.ini_path != nullptr) {
            Matrix<uint8_t> load_dets(max_n_dets, det_size);
            double *load_vals = sol_vec.values();
            
            size_t n_dets = load_vec_txt(*args.ini_path, load_dets, load_vals);
            
            for (size_t det_idx = 0; det_idx < n_dets; det_idx++) {
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
        sol_vec.perform_add(0);
        loc_norm = sol_vec.local_norm();
        glob_norm = sum_mpi(loc_norm, proc_rank, n_procs);
        if (args.load_dir != nullptr) {
            last_norm = glob_norm;
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
            file_path.append("nini.txt");
            ini_file.open(file_path, std::ofstream::app);
            
            file_path = args.result_dir;
            file_path.append("params.txt");
            std::ofstream param_f(file_path);
            param_f << "FRI calculation\nFCIDUMP path: " << args.fcidump_path << "\nepsilon (imaginary time step): " << eps << "\nTarget norm " << target_norm << "\nInitiator threshold: " << args.init_thresh << "\nMatrix nonzero: " << args.matr_samp << "\nVector nonzero: " << args.target_nonz << "\n";
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
        
        hb_info *hb_probs = set_up(tot_orb, n_orb, *eris);
        
        double last_one_norm = 0;
        
        // Parameters for systematic sampling
        double rn_sys = 0;
        int glob_n_nonz; // Number of nonzero elements in whole vector (across all processors)
        double loc_norms[n_procs];
        max_n_dets = (unsigned int)sol_vec.max_size();
        std::vector<size_t> srt_arr(max_n_dets);
        std::vector<bool> keep_exact(max_n_dets, false);
        
#pragma mark Pre-calculate deterministic subspace of Hamiltonian
        size_t determ_h_size = n_determ * n_elec_unf * n_elec_unf * (n_orb - n_elec_unf / 2) * (n_orb - n_elec_unf / 2);
        unsigned int n_determ_h = 0;
        size_t *determ_from = (size_t *)malloc(determ_h_size * sizeof(size_t));
        Matrix<uint8_t> determ_to(determ_h_size, det_size);
        double *determ_matr_el = (double *)malloc(determ_h_size * sizeof(double));
        for (size_t det_idx = 0; det_idx < n_determ; det_idx++) {
            uint8_t *curr_det = sol_vec.indices()[det_idx];
            uint8_t *occ_orbs = sol_vec.orbs_at_pos(det_idx);
            uint8_t (*sing_ex_orbs)[2] = (uint8_t (*)[2])comp_vecs.orb_indices1;
            size_t ex_idx;
            
            size_t n_sing = sing_ex_symm(curr_det, occ_orbs, n_elec_unf, n_orb, sing_ex_orbs, symm);
            if (n_sing + n_determ_h > determ_h_size) {
                std::cout << "Allocating more memory for deterministic part of Hamiltonian\n";
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
            
            uint8_t (*doub_ex_orbs)[4] = (uint8_t (*)[4])comp_vecs.orb_indices1;
            size_t n_doub = doub_ex_symm(curr_det, occ_orbs, n_elec_unf, n_orb, doub_ex_orbs, symm);
            if (n_doub + n_determ_h > determ_h_size) {
                std::cout << "Allocating more memory for deterministic part of Hamiltonian\n";
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
            std::cout << "Elements in dense H: " << tot_dense_h << "\n";
        }
        
        unsigned int iterat;
        size_t n_ini;
        for (iterat = 0; iterat < args.max_iter; iterat++) {
            n_ini = 0;
            glob_n_nonz = sum_mpi(sol_vec.n_nonz(), proc_rank, n_procs);
            
            // Systematic matrix compression
            if (glob_n_nonz > args.matr_samp) {
                std::cerr << "Warning: target number of matrix samples " << args.matr_samp << " is less than number of nonzero vector elements (" << glob_n_nonz << ")\n";
            }
            
            std::copy(sol_vec.values() + n_determ, sol_vec.values() + sol_vec.curr_size(), comp_vecs.vec1.begin());
            for (size_t det_idx = n_determ; det_idx < sol_vec.curr_size(); det_idx++) {
                comp_vecs.det_indices1[det_idx - n_determ] = det_idx;
            }

            size_t comp_len = sol_vec.curr_size() - n_determ;
            comp_vecs.vec_len = comp_len;
            apply_HBPP_sys(sol_vec.occ_orbs(), sol_vec.indices(), &comp_vecs, hb_probs, &basis_symm, p_doub, new_hb, mt_obj, matr_samp - tot_dense_h, sing_shortcut, doub_shortcut);
            comp_len = comp_vecs.vec_len;
            
            double *vals_before_mult = sol_vec.values();
            sol_vec.set_curr_vec_idx(1);
            sol_vec.zero_vec();
            size_t vec_size = sol_vec.curr_size();
            
            // The first time around, add only elements that came from noninitiators
            for (int add_ini = 0; add_ini < 2; add_ini++) {
                int num_added = 1;
                size_t samp_idx = 0;
                while (num_added > 0) {
                    num_added = 0;
                    while (samp_idx < comp_len && num_added < adder_size) {
                        size_t det_idx = comp_vecs.det_indices2[samp_idx];
                        double curr_val = vals_before_mult[det_idx];
                        uint8_t ini_flag = fabs(curr_val) >= args.init_thresh;
                        if (ini_flag != add_ini) {
                            samp_idx++;
                            continue;
                        }
                        uint8_t *curr_det = sol_vec.indices()[det_idx];
                        uint8_t new_det[det_size];
                        
                        double add_el = -eps * comp_vecs.vec1[samp_idx];
                        if (curr_val < 0) {
                            add_el *= -1;
                        }
                        std::copy(curr_det, curr_det + det_size, new_det);
                        if (!(comp_vecs.orb_indices1[samp_idx][2] == 0 && comp_vecs.orb_indices1[samp_idx][3] == 0)) { // double excitation
                            uint8_t *doub_orbs = comp_vecs.orb_indices1[samp_idx];
                            doub_det(new_det, doub_orbs);
                        }
                        else { // single excitation
                            uint8_t *sing_orbs = comp_vecs.orb_indices1[samp_idx];
                            sing_det(new_det, sing_orbs);
                        }
                        sol_vec.add(new_det, add_el, ini_flag);
                        num_added++;
                        samp_idx++;
                    }
                    sol_vec.perform_add(0);
                    sol_vec.set_curr_vec_idx(0);
                    vals_before_mult = sol_vec.values();
                    sol_vec.set_curr_vec_idx(1);
                    num_added = sum_mpi(num_added, proc_rank, n_procs);
                }
            }
            size_t new_max_dets = sol_vec.max_size();
            if (new_max_dets > max_n_dets) {
                keep_exact.resize(new_max_dets, false);
                srt_arr.resize(new_max_dets);
                max_n_dets = new_max_dets;
            }
            
#pragma mark Perform deterministic subspace multiplication
            for (size_t samp_idx = 0; samp_idx < determ_h_size; samp_idx++) {
                size_t det_idx = determ_from[samp_idx];
                double mat_vec = vals_before_mult[det_idx] * determ_matr_el[samp_idx];
                sol_vec.add(determ_to[samp_idx], mat_vec, 1);
            }
            sol_vec.perform_add(0);
            
#pragma mark Death/cloning step
            sol_vec.set_curr_vec_idx(0);
            for (size_t det_idx = 0; det_idx < vec_size; det_idx++) {
                double *curr_val = sol_vec[det_idx];
                if (*curr_val != 0) {
                    double diag_el = sol_vec.matr_el_at_pos(det_idx);
                    *curr_val *= 1 - eps * (diag_el - en_shift);
                }
            }
            sol_vec.add_vecs(0, 1);
            sol_vec.set_curr_vec_idx(1);
            sol_vec.zero_vec();
            sol_vec.set_curr_vec_idx(0);
            
#pragma mark Vector compression step
            unsigned int n_samp = args.target_nonz;
            loc_norms[proc_rank] = find_preserve(&(sol_vec.values()[n_determ]), srt_arr, keep_exact, sol_vec.curr_size() - n_determ, &n_samp, &glob_norm);
            glob_norm += sol_vec.dense_norm();
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
            double numer = sol_vec.dot(htrial_vec.indices(), htrial_vec.values(), htrial_vec.curr_size(), htrial_hashes);
            double denom = sol_vec.dot(trial_vec.indices(), trial_vec.values(), trial_vec.curr_size(), trial_hashes);
            numer = sum_mpi(numer, proc_rank, n_procs);
            denom = sum_mpi(denom, proc_rank, n_procs);
            if (proc_rank == hf_proc) {
                num_file << numer << '\n';
                den_file << denom << '\n';
                std::cout << iterat << ", en est: " << numer / denom << ", shift: " << en_shift << ", norm: " << glob_norm << '\n';
                ini_file << n_ini << '\n';
            }
            
            if (proc_rank == 0) {
                rn_sys = mt_obj() / (1. + UINT32_MAX);
            }

            MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, loc_norms, 1, MPI_DOUBLE, MPI_COMM_WORLD);
            sys_comp(&(sol_vec.values()[n_determ]), sol_vec.curr_size() - n_determ, loc_norms, n_samp, keep_exact, rn_sys);
            for (size_t det_idx = 0; det_idx < sol_vec.curr_size() - n_determ; det_idx++) {
                if (keep_exact[det_idx]) {
                    sol_vec.del_at_pos(det_idx + n_determ);
                    keep_exact[det_idx] = 0;
                }
            }
            
            if ((iterat + 1) % save_interval == 0) {
                sol_vec.save(args.result_dir);
                uint64_t tot_add = sol_vec.tot_sgn_coh();
                if (proc_rank == hf_proc) {
                    num_file.flush();
                    den_file.flush();
                    shift_file.flush();
                    nkept_file.flush();
                    std::cout << "Total additions to nonzero: " << tot_add << "\n";
                }
            }
        }
        sol_vec.save(args.result_dir);
        if (proc_rank == hf_proc) {
            num_file.close();
            den_file.close();
            shift_file.close();
            nkept_file.close();
        }

        MPI_Finalize();
    } catch (std::exception &ex) {
        std::cerr << "\nException : " << ex.what() << "\n\nPlease send a description of this error, a copy of the command-line arguments used, and the random number generator seeds printed for each process to the developers through our GitHub repository: https://github.com/sgreene8/FRIES/ \n\n";
    }
    return 0;
}

