
/*! \file
 *
 * \brief FRI with unnormalized HB-PP matrix factorization applied to subspace iteration for calculating excited-state energies.
 * Attempts to propagate the solution separately in the trial subspace (without compression error) and in the space orthogonal to it
 * Found to yield less stochastic error than standard subspace iteration for small systems but more error for larger systems (C2 in cc-pVQZ)
 */

#include <iostream>
#include <chrono>
#include <FRIES/io_utils.hpp>
#include <FRIES/compress_utils.hpp>
#include <FRIES/Ext_Libs/argparse.hpp>
#include <FRIES/Hamiltonians/near_uniform.hpp>
#include <FRIES/Hamiltonians/heat_bathPP.hpp>
#include <FRIES/Hamiltonians/molecule.hpp>
#include <FRIES/fci_utils.h>
#include <LAPACK/lapack_wrappers.hpp>
#include <FRIES/Ext_Libs/cnpy/cnpy.h>
#include <stdexcept>
#include <iomanip>

struct MyArgs : public argparse::Args {
    std::string &hf_path = kwarg("hf_path", "Path to the directory that contains the HF output files eris.txt, hcore.txt, symm.txt, hf_en.txt, and sys_params.txt");
    uint32_t &max_iter = kwarg("max_iter", "Maximum number of iterations to run the calculation").set_default(1000000);
    uint32_t &target_nonz = kwarg("vec_nonz", "Target number of nonzero vector elements to keep after each iteration");
    std::string &result_dir = kwarg("result_dir", "Directory in which to save output files").set_default<std::string>("./");
    uint32_t &max_n_dets = kwarg("max_dets", "Maximum number of determinants on a single MPI process");
    std::string &trial_path = kwarg("trial_vecs", "Path to files containing the vectors with which to calculate the energy and initialize the calculation. If the suffix is '.dice', FRIES will treat the file as output from a Dice calculation and read in the vectors accordingly. Otherwise, FRIES will look for text files with names <trial_vecs>dets<xx> and <trial_vecs>vals<xx>, where xx is a 2-digit number ranging from 0 to (num_trial - 1).");
    uint32_t &n_trial = kwarg("num_trial", "Number of trial vectors to use to calculate dot products with the iterates");
    uint32_t &restart_int = kwarg("restart_int", "Number of multiplications by (1 - \eps H) to do in between restarts").set_default(1000);
    std::string &mat_output = kwarg("out_format", "A flag controlling the format for outputting the Hamiltonian and overlap matrices. Must be either 'none', in which case these matrices are not written to disk, 'txt', in which case they are outputted in text format, 'npy', in which case they are outputted in numpy format, or 'bin' for binary format").set_default<std::string>("none");
    double &init_thresh = kwarg("initiator", "Magnitude of vector element required to make it an initiator").set_default(0);
    std::shared_ptr<std::string> &load_dir = kwarg("load_dir", "Directory from which to load checkpoint files from a previous FRI calculation (in binary format, see documentation for DistVec::save() and DistVec::load())");
    int &time_reversal = kwarg("time_reversal", "0 if time-reversal symmetry is not to be used, 1 for time-reversal symmetry with an even-spin state, -1 for time-reversal symmetry with an odd-spin state").set_default(0);
    double &epsilon = kwarg("epsilon", "The imaginary time step (\eps) to use when evolving the vectors. Overrides the parameter listed in sys_params.txt").set_default(0);
};

int main(int argc, char * argv[]) {
    MyArgs args = argparse::parse<MyArgs>(argc, argv);
    
    try {
        int n_procs = 1;
        int proc_rank = 0;
        unsigned int hf_proc;
        
        MPI_Init(NULL, NULL);
        MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
        
        size_t max_n_dets = args.max_n_dets;
        uint32_t matr_samp = args.target_nonz;
        uint8_t n_trial = args.n_trial;
        
        // Parameters
        double shift_damping = 0.5;
        unsigned int shift_interval = 1;
        unsigned int save_interval = 1000;
        
        enum {
            no_out,
            txt_out,
            npy_out,
            bin_out
        } mat_fmt;
        
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
        if (args.time_reversal < -1 || args.time_reversal > 1) {
            throw std::runtime_error("time_reversal input argument must be -1, 0, or 1");
        }
        
        // Read in data files
        fcidump_input *in_data = parse_fcidump(args.hf_path);
        double eps = in_data->eps;
        if (args.epsilon > 0) {
            eps = args.epsilon;
        }
        unsigned int n_elec = in_data->n_elec;
        unsigned int n_frz = 0;
        unsigned int n_orb = in_data->n_orb_;
        double hf_en = in_data->hf_en;
        
        unsigned int n_elec_unf = n_elec - n_frz;
        unsigned int tot_orb = n_orb + n_frz / 2;
        
        uint8_t *symm = in_data->symm;
        Matrix<double> *h_core = in_data->hcore;
        SymmERIs *eris = &(in_data->eris);
        
        // Rn generator
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::cout << "seed on process " << proc_rank << " is " << seed << std::endl;
        std::mt19937 mt_obj((unsigned int)seed);
        
        unsigned int spawn_length = matr_samp * 4 / n_procs;
        size_t adder_size = spawn_length > 1000000 ? 1000000 : spawn_length;
        std::function<double(const uint8_t *)> diag_shortcut;
        int time_reversal = args.time_reversal;
        if (time_reversal) {
            diag_shortcut = [tot_orb, eris, h_core, n_frz, n_elec, hf_en, time_reversal](const uint8_t *occ_orbs) {
                double ref_matrel = diag_matrel(occ_orbs, tot_orb, *eris, *h_core, n_frz, n_elec) - hf_en;
                uint8_t idx[4];
                uint32_t n_orb = tot_orb - n_frz / 2;
                int doub_connect = tr_doub_connect(occ_orbs, n_orb, n_elec - n_frz, idx);
                if (doub_connect == 0) { // equal to flipped-spin determinant
                    if (time_reversal == -1) {
                        ref_matrel = 0;
                    }
                }
                else if (doub_connect == 1) {
                    uint32_t half_elec = (n_elec - n_frz) / 2;
                    idx[2] = occ_orbs[idx[1]] - n_orb;
                    idx[3] = occ_orbs[idx[0]] + n_orb;
                    int sign = excite_sign_occ(idx[0], idx[2], occ_orbs, half_elec);
                    sign *= excite_sign_occ(idx[1] - half_elec, idx[3], occ_orbs + half_elec, half_elec);
                    
                    idx[0] = occ_orbs[idx[0]];
                    idx[1] = occ_orbs[idx[1]];
                    double doub_matrel = doub_matr_el_nosgn(idx, tot_orb, *eris, n_frz) * sign;
                    ref_matrel += time_reversal * doub_matrel;
                }
                return ref_matrel;
            };
        }
        else {
            diag_shortcut = [tot_orb, eris, h_core, n_frz, n_elec, hf_en](const uint8_t *occ_orbs) {
                return diag_matrel(occ_orbs, tot_orb, *eris, *h_core, n_frz, n_elec) - hf_en;
            };
        }
        std::function<double(uint8_t *, uint8_t *)> sing_shortcut = [tot_orb, eris, h_core, n_frz, n_elec_unf](uint8_t *ex_orbs, uint8_t *occ_orbs) {
            return sing_matr_el_nosgn(ex_orbs, occ_orbs, tot_orb, *eris, *h_core, n_frz, n_elec_unf);
        };
        std::function<double(uint8_t *)> doub_shortcut = [tot_orb, eris, n_frz](uint8_t *ex_orbs) {
            return doub_matr_el_nosgn(ex_orbs, tot_orb, *eris, n_frz);
        };
        size_t det_size = CEILING(2 * n_orb, 8);
        
        SymmInfo basis_symm(in_data->symm, n_orb);
        
        // Initialize hash function for processors and vector
        std::vector<uint32_t> proc_scrambler(2 * n_orb);
        
        if (args.load_dir != nullptr) {
            load_proc_hash(*args.load_dir, proc_scrambler.data());
        }
        else {
            if (proc_rank == 0) {
                for (size_t proc_idx = 0; proc_idx < 2 * n_orb; proc_idx++) {
                    proc_scrambler[proc_idx] = mt_obj();
                }
                save_proc_hash(args.result_dir, proc_scrambler.data(), 2 * n_orb);
            }
            
            MPI_Bcast(proc_scrambler.data(), 2 * n_orb, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        }
        std::vector<uint32_t> vec_scrambler(2 * n_orb);
        for (size_t proc_idx = 0; proc_idx < 2 * n_orb; proc_idx++) {
            vec_scrambler[proc_idx] = mt_obj();
        }
        
        DistVec<double> sol_vec(args.max_n_dets, adder_size, n_orb * 2, n_elec_unf, n_procs, diag_shortcut, 2 * n_trial, proc_scrambler, vec_scrambler);
        
        size_t n_states = n_elec_unf > (n_orb - n_elec_unf / 2) ? n_elec_unf : n_orb - n_elec_unf / 2;
        HBCompressPiv comp_vecs(spawn_length, n_states);
        
        uint8_t hf_det[det_size];
        gen_hf_bitstring(n_orb, n_elec - n_frz, hf_det);
        hf_proc = sol_vec.idx_to_proc(hf_det);
        
        uint8_t tmp_orbs[n_elec_unf];
        
        std::string dice_ext(".dice");
        bool dice_input = std::equal(args.trial_path.end() - 5, args.trial_path.end(), dice_ext.begin());
        
# pragma mark Set up trial vectors
        std::stringstream tmp_path;
        
        Matrix<uint8_t> *load_dets = new Matrix<uint8_t>(args.max_n_dets, det_size);
        sol_vec.set_curr_vec_idx(n_trial);
        double *load_vals = sol_vec.values();
        for (uint8_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
            size_t loc_n_dets;
            if (dice_input) {
                loc_n_dets = load_vec_dice(args.trial_path, *load_dets, load_vals, trial_idx, n_orb);
            }
            else {
                tmp_path << args.trial_path << std::setfill('0') << std::setw(2) << (int) trial_idx;
                loc_n_dets = load_vec_txt(tmp_path.str(), *load_dets, load_vals);
            }
            
            for (size_t det_idx = 0; det_idx < loc_n_dets; det_idx++) {
                uint8_t flipped_det[det_size];
                uint8_t *new_det = (*load_dets)[det_idx];
                uint8_t *ref_det;
                if (time_reversal) {
                    flip_spins(new_det, flipped_det, n_orb);
                    int cmp = memcmp(new_det, flipped_det, det_size);
                    if (cmp > 0) {
                        ref_det = flipped_det;
                        load_vals[det_idx] *= time_reversal;
                    }
                    else {
                        ref_det = new_det;
                    }
                    if (cmp != 0) {
                        load_vals[det_idx] /= sqrt(2);
                    }
                }
                else {
                    ref_det = new_det;
                }
                sol_vec.add(ref_det, load_vals[det_idx], 1);
            }
            sol_vec.set_curr_vec_idx(trial_idx);
            sol_vec.perform_add(0);
            tmp_path.str("");
        }
        
        delete load_dets;
        size_t trial_size = sol_vec.curr_size();
        
        std::cout << "Vector size after trial vectors: " << trial_size << '\n';
        
# pragma mark Orthonormalize trial vectors
        for (uint8_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
            for (uint8_t orthvec_idx = 0; orthvec_idx < trial_idx; orthvec_idx++) {
                double dprod = sol_vec.internal_dot(orthvec_idx, trial_idx);
                dprod = sum_mpi(dprod, proc_rank, n_procs);
                for (size_t el_idx = 0; el_idx < trial_size; el_idx++) {
                    *sol_vec[el_idx] -= dprod * *sol_vec(orthvec_idx, el_idx);
                }
            }
            sol_vec.set_curr_vec_idx(trial_idx);
            double two_norm = sol_vec.two_norm();
            two_norm = sum_mpi(two_norm, proc_rank, n_procs);
            for (size_t el_idx = 0; el_idx < trial_size; el_idx++) {
                *sol_vec[el_idx] /= sqrt(two_norm);
            }
        }
        
        Matrix<double> trial_vals(n_trial, trial_size);
        for (uint8_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
            std::copy(sol_vec(trial_idx, 0), sol_vec(trial_idx, trial_size), trial_vals[trial_idx]);
        }
        
        for (uint8_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
            sol_vec.set_curr_vec_idx(trial_idx);
            h_op_diag(sol_vec, trial_idx + n_trial, 0, 1);
            sol_vec.set_curr_vec_idx(trial_idx);
            h_op_offdiag(sol_vec, trial_size, symm, tot_orb, *eris, *h_core, (uint8_t *)comp_vecs.orb_indices1, 4 * spawn_length, n_frz, n_elec_unf, trial_idx + n_trial, 1, time_reversal);
        }
        size_t htrial_size = sol_vec.curr_size();
        Matrix<double> htrial_vals(n_trial, htrial_size);
        for (uint8_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
            sol_vec.set_curr_vec_idx(n_trial + trial_idx);
            std::copy(sol_vec.values(), sol_vec.values() + htrial_size, htrial_vals[trial_idx]);
            sol_vec.zero_vec();
            sol_vec.set_curr_vec_idx(trial_idx);
            sol_vec.zero_vec();
        }
        
        std::cout << "Vector size after H multiplication: " << htrial_size << '\n';
        
        sol_vec.fix_min_del_idx();
        
        Matrix<double> uhu(n_trial, n_trial); // trans(trials) * H * trials
        for (uint8_t vec_idx1 = 0; vec_idx1 < n_trial; vec_idx1++) {
            for (uint8_t vec_idx2 = 0; vec_idx2 < n_trial; vec_idx2++) {
                double dprod = 0;
                for (size_t el_idx = 0; el_idx < trial_size; el_idx++) {
                    dprod += trial_vals(vec_idx1, el_idx) * htrial_vals(vec_idx2, el_idx);
                }
                dprod = sum_mpi(dprod, proc_rank, n_procs);
                uhu(vec_idx1, vec_idx2) = dprod;
            }
        }
        
        // Count # single/double excitations from HF
        sol_vec.gen_orb_list(hf_det, tmp_orbs);
        size_t n_hf_doub = doub_ex_symm(hf_det, tmp_orbs, n_elec_unf, n_orb, comp_vecs.orb_indices1, symm);
        size_t n_hf_sing = count_singex(hf_det, tmp_orbs, n_elec_unf, &basis_symm);
        double p_doub = (double) n_hf_doub / (n_hf_sing + n_hf_doub);
        
        std::string file_path;
        std::ofstream bmat_file;
        std::ofstream dmat_file;
        
        if (proc_rank == 0) {
            if (mat_fmt == txt_out) {
                tmp_path << args.result_dir << "h_mat.txt";
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
                tmp_path << args.result_dir << "h_mat.dat";
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
            param_f << "Subspace iteration\nHF path: " << args.hf_path << "\nepsilon (imaginary time step): " << eps << "\nVector nonzero: " << args.target_nonz << "\nInitiator threshold: " << args.init_thresh << "\n";
            param_f << "Path for trial vectors: " << args.trial_path << "\n";
            param_f << "Restart interval: " << args.restart_int << "\n";
            if (args.load_dir != nullptr) {
                param_f << "Restarting calculation from " << args.load_dir << "\n";
            }
            param_f.close();
        }
        
        hb_info *hb_probs = set_up(tot_orb, n_orb, *eris);
        
        // Vectors for systematic sampling
        if (sol_vec.max_size() > max_n_dets) {
            max_n_dets = sol_vec.max_size();
        }
        std::vector<size_t> srt_arr(max_n_dets);
        std::vector<bool> keep_exact(max_n_dets, false);
        std::vector<bool> del_all(max_n_dets, true);
        
        Matrix<double> h_mat(n_trial, n_trial);
        tmp_path.str("");
        tmp_path << args.result_dir << "d_mat.npy";
        std::string dnpy_path(tmp_path.str());
        tmp_path.str("");
        tmp_path << args.result_dir << "h_mat.npy";
        std::string hnpy_path(tmp_path.str());
        tmp_path.str("");
        tmp_path << args.result_dir << "shifts.txt";
        std::ofstream shift_f(tmp_path.str());
        tmp_path.str("");
        tmp_path << args.result_dir << "norms.txt";
        std::ofstream norm_f(tmp_path.str());
        tmp_path.str("");
        tmp_path << args.result_dir << "n_ini.txt";
        std::ofstream nini_f(tmp_path.str());
        
        std::vector<double> lapack_scratch((3 + n_trial + 32 * 2) * n_trial - 1);
        Matrix<double> lapack_inout(n_trial, n_trial);
        std::vector<double> norms(n_trial);
        std::vector<double> last_norms(n_trial);
        
        int vec_half = 0; // controls whether the current iterates are stored in the first or second half of the values_ matrix
        
        std::vector<double> en_shifts(n_trial);
        Matrix<double> trial_iterat(n_trial, n_trial); // matrix defining the contribution of the trial vectors to the current iterate
        for (uint8_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
            trial_iterat(vec_idx, vec_idx) = 1000;
        }
        Matrix<double> uhum(n_trial, n_trial); // uhu * trial_iterate
        Matrix<double> tmp_mat(n_trial, n_trial);
        
        for (uint32_t iteration = 0; iteration < args.max_iter; iteration++) {
            if (proc_rank == 0) {
                std::cout << "Iteration " << iteration << "\n";
            }
#pragma mark Normalize vectors
            for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                sol_vec.set_curr_vec_idx(vec_half * n_trial + vec_idx);
                double norm = 0;
                for (size_t el_idx = 0; el_idx < trial_size; el_idx++) {
                    double element = *sol_vec[el_idx];
                    for (uint8_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
                        element += trial_vals(trial_idx, el_idx) * trial_iterat(trial_idx, vec_idx);
                    }
                    norm += fabs(element);
                }
                for (size_t el_idx = trial_size; el_idx < sol_vec.curr_size(); el_idx++) {
                    norm += fabs(*sol_vec[el_idx]);
                }
                norms[vec_idx] = sum_mpi(norm, proc_rank, n_procs);
            }

            if (iteration == 0) {
                std::copy(norms.begin(), norms.end(), last_norms.begin());
                std::fill(en_shifts.begin(), en_shifts.end(), 1);
            }
            if ((iteration + 1) % shift_interval == 0) {
                for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                    adjust_shift2(&en_shifts[vec_idx], norms[vec_idx], &last_norms[vec_idx], shift_damping);
                }
                if (proc_rank == 0) {
                    for (uint16_t vec_idx = 0; vec_idx < n_trial - 1; vec_idx++) {
                        shift_f << en_shifts[vec_idx] << ',';
                    }
                    shift_f << en_shifts[n_trial - 1] << '\n';
                    shift_f.flush();
                }
            }
            if (proc_rank == 0) {
                for (uint16_t vec_idx = 0; vec_idx < n_trial - 1; vec_idx++) {
                    norm_f << norms[vec_idx] << ',';
                }
                norm_f << norms[n_trial - 1] << '\n';
                norm_f.flush();
            }
            for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                sol_vec.set_curr_vec_idx(vec_half * n_trial + vec_idx);
                for (size_t el_idx = 0; el_idx < sol_vec.curr_size(); el_idx++) {
                    *sol_vec[el_idx] /= en_shifts[vec_idx];
                }
                for (uint8_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
                    trial_iterat(trial_idx, vec_idx) /= en_shifts[vec_idx];
                }
            }
            
#pragma mark Calculate overlap and hamiltonian matrices
            uhum.zero();
            for (uint16_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
                for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                    for (uint16_t sum_idx = 0; sum_idx < n_trial; sum_idx++) {
                        uhum(trial_idx, vec_idx) += uhu(trial_idx, sum_idx) * trial_iterat(sum_idx, vec_idx);
                    }
                    double d_prod = 0;
                    for (size_t det_idx = 0; det_idx < htrial_size; det_idx++) {
                        d_prod += htrial_vals(trial_idx, det_idx) * (*sol_vec(vec_idx + vec_half * n_trial, det_idx));
                    }
                    d_prod = sum_mpi(d_prod, proc_rank, n_procs);
                    h_mat(trial_idx, vec_idx) = d_prod + uhum(trial_idx, vec_idx);
                }
            }
            
            if (proc_rank == 0) {
                if (mat_fmt == npy_out) {
                    cnpy::npy_save(hnpy_path, h_mat.data(), {1, n_trial, n_trial}, "a");
                    cnpy::npy_save(dnpy_path, trial_iterat.data(), {1, n_trial, n_trial}, "a");
                }
                else if (mat_fmt == txt_out) {
                    for (uint16_t row_idx = 0; row_idx < n_trial; row_idx++) {
                        for (uint16_t col_idx = 0; col_idx < n_trial - 1; col_idx++) {
                            bmat_file << h_mat(row_idx, col_idx) << ",";
                            dmat_file << trial_iterat(row_idx, col_idx) << ",";
                        }
                        bmat_file << h_mat(row_idx, n_trial - 1) << "\n";
                        dmat_file << trial_iterat(row_idx, n_trial - 1) << "\n";
                    }
                    dmat_file.flush();
                    bmat_file.flush();
                }
                else if (mat_fmt == bin_out) {
                    bmat_file.write((char *) h_mat.data(), sizeof(double) * n_trial * n_trial);
                    dmat_file.write((char *) trial_iterat.data(), sizeof(double) * n_trial * n_trial);
                    bmat_file.flush();
                    dmat_file.flush();
                }
            }
            
#pragma mark Orthogonalization
            if ((iteration + 1) % args.restart_int == 0) {
                for (size_t a = 0; a < n_trial; a++) {
                    for (size_t b = 0; b < n_trial; b++) {
                        lapack_inout(a, b) = trial_iterat(a, b) - eps * h_mat(a, b);
                    }
                }
                invr_inplace(lapack_inout, lapack_scratch.data());
                for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                    sol_vec.set_curr_vec_idx(vec_half * n_trial + vec_idx);
                    double norm = 0;
                    for (size_t el_idx = 0; el_idx < trial_size; el_idx++) {
                        double element = *sol_vec[el_idx];
                        for (uint8_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
                            element += trial_vals(trial_idx, el_idx) * trial_iterat(trial_idx, vec_idx);
                        }
                        norm += fabs(element);
                    }
                    for (size_t el_idx = trial_size; el_idx < sol_vec.curr_size(); el_idx++) {
                        norm += fabs(*sol_vec[el_idx]);
                    }
                    norms[vec_idx] = sum_mpi(norm, proc_rank, n_procs);
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
                tmp_mat.zero();
                for (uint8_t eigen_idx = 0; eigen_idx < n_trial; eigen_idx++) {
                    for (uint8_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                        for (uint8_t el_idx = 0; el_idx < n_trial; el_idx++) {
                            tmp_mat(el_idx, eigen_idx) += trial_iterat(el_idx, vec_idx) * lapack_inout(vec_idx, eigen_idx);
                        }
                    }
                }
                trial_iterat.copy_from(tmp_mat);

                tmp_mat.zero();
                for (uint8_t eigen_idx = 0; eigen_idx < n_trial; eigen_idx++) {
                    for (uint8_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                        for (uint8_t el_idx = 0; el_idx < n_trial; el_idx++) {
                            tmp_mat(el_idx, eigen_idx) += h_mat(el_idx, vec_idx) * lapack_inout(vec_idx, eigen_idx);
                        }
                    }
                }
                h_mat.copy_from(tmp_mat);

                vec_half = !vec_half;
                for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                    sol_vec.set_curr_vec_idx(vec_half * n_trial + vec_idx);
                    double new_norm = 0;
                    for (size_t el_idx = 0; el_idx < trial_size; el_idx++) {
                        double element = *sol_vec[el_idx];
                        for (uint8_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
                            element += trial_vals(trial_idx, el_idx) * trial_iterat(trial_idx, vec_idx);
                        }
                        new_norm += fabs(element);
                    }
                    for (size_t el_idx = trial_size; el_idx < sol_vec.curr_size(); el_idx++) {
                        new_norm += fabs(*sol_vec[el_idx]);
                    }
                    new_norm = sum_mpi(new_norm, proc_rank, n_procs);
                    for (size_t el_idx = 0; el_idx < sol_vec.curr_size(); el_idx++) {
                        *sol_vec[el_idx] *= norms[vec_idx] / new_norm;
                    }
                    for (uint8_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
                        trial_iterat(trial_idx, vec_idx) *= norms[vec_idx] / new_norm;
                        h_mat(trial_idx, vec_idx) *= norms[vec_idx] / new_norm;
                    }
                }
                
                uhum.zero();
                for (uint16_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
                    for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                        for (uint16_t sum_idx = 0; sum_idx < n_trial; sum_idx++) {
                            uhum(trial_idx, vec_idx) += uhu(trial_idx, sum_idx) * trial_iterat(sum_idx, vec_idx);
                        }
                    }
                }
            }
            
#pragma mark Vector compression
            compress_vecs(sol_vec, vec_half * n_trial, (vec_half + 1) * n_trial, matr_samp, srt_arr, keep_exact, del_all, mt_obj);
            
# pragma mark Matrix multiplication
            for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                int curr_idx = vec_half * n_trial + vec_idx;
                int next_idx = (1 - vec_half) * n_trial + vec_idx;
                sol_vec.set_curr_vec_idx(curr_idx);

                double one_norm = 0;
                for (size_t el_idx = 0; el_idx < trial_size; el_idx++) {
                    double element = *sol_vec[el_idx];
                    for (uint8_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
                        element += trial_vals(trial_idx, el_idx) * trial_iterat(trial_idx, vec_idx);
                    }
                    one_norm += fabs(element);
                }
                for (size_t el_idx = trial_size; el_idx < sol_vec.curr_size(); el_idx++) {
                    one_norm += fabs(*sol_vec[el_idx]);
                }
                one_norm = sum_mpi(one_norm, proc_rank, n_procs);
                double init_thresh = args.init_thresh * one_norm / matr_samp;

                size_t comp_len = 0;
                for (size_t det_idx = 0; det_idx < sol_vec.curr_size(); det_idx++) {
                    if (sol_vec.values()[det_idx] != 0) {
                        double value = sol_vec.values()[det_idx];
                        if (det_idx < trial_size) {
                            for (uint8_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
                                value += trial_vals(trial_idx, det_idx) * trial_iterat(trial_idx, vec_idx);
                            }
                        }
                        comp_vecs.vec1[comp_len] = value;
                        comp_vecs.det_indices1[comp_len] = det_idx;
                        comp_len++;
                    }
                }
                if (comp_len > spawn_length) {
                    std::cerr << "Error: insufficient memory allocated for matrix compression.\n";
                }
                comp_vecs.vec_len = comp_len;
                apply_HBPP_piv(sol_vec.occ_orbs(), sol_vec.indices(), &comp_vecs, hb_probs, &basis_symm, p_doub, true, mt_obj, matr_samp, sing_shortcut, doub_shortcut, time_reversal);
                comp_len = comp_vecs.vec_len;

                double *vals_before_mult = sol_vec.values();
                sol_vec.set_curr_vec_idx(next_idx);
                sol_vec.zero_vec();
                size_t vec_size = sol_vec.curr_size();

                // The first time around, add only elements that came from noninitiators
                for (int add_ini = 0; add_ini < 2; add_ini++) {
                    int num_added = 1;
                    size_t samp_idx = 0;
                    while (num_added > 0) {
                        num_added = 0;
                        uint8_t flip_det[det_size];
                        while (samp_idx < comp_len && num_added < adder_size) {
                            size_t det_idx = comp_vecs.det_indices2[samp_idx];
                            double curr_val = vals_before_mult[det_idx];
//                            int neg = curr_val < 0;
                            if (det_idx < trial_size) {
                                for (uint8_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
                                    curr_val += trial_vals(trial_idx, det_idx) * trial_iterat(trial_idx, vec_idx);
                                }
                            }
                            uint8_t ini_flag = fabs(curr_val) >= init_thresh;
                            if (ini_flag != add_ini) {
                                samp_idx++;
                                continue;
                            }
                            uint8_t *curr_det = sol_vec.indices()[det_idx];
                            uint8_t new_det[det_size];
                            double add_el = -eps * comp_vecs.vec1[samp_idx];// * vals_before_mult[det_idx] / fabs(curr_val);
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
                            uint8_t *ref_det;
                            if (time_reversal) {
                                flip_spins(new_det, flip_det, n_orb);
                                if (memcmp(new_det, flip_det, det_size) > 0) {
                                    ref_det = flip_det;
                                }
                                else {
                                    ref_det = new_det;
                                }
                            }
                            else {
                                ref_det = new_det;
                            }
                            sol_vec.add(ref_det, add_el, ini_flag);
                            num_added++;
                            samp_idx++;
                        }
                        sol_vec.perform_add(curr_idx);
                        sol_vec.set_curr_vec_idx(curr_idx);
                        vals_before_mult = sol_vec.values();
                        sol_vec.set_curr_vec_idx(next_idx);
                        num_added = sum_mpi(num_added, proc_rank, n_procs);
                    }
                }
                size_t new_max_dets = sol_vec.max_size();
                if (new_max_dets > max_n_dets) {
                    keep_exact.resize(new_max_dets, false);
                    srt_arr.resize(new_max_dets);
                    del_all.resize(new_max_dets);
                    max_n_dets = new_max_dets;
                }
                
                sol_vec.set_curr_vec_idx(curr_idx);
                int ini_count = 0;
                for (size_t det_idx = 0; det_idx < vec_size; det_idx++) {
                    double curr_val = *(sol_vec[det_idx]);
                    if (det_idx < trial_size) {
                        for (uint8_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
                            curr_val += trial_vals(trial_idx, det_idx) * trial_iterat(trial_idx, vec_idx);
                        }
                    }
                    if (curr_val != 0) {
                        ini_count += fabs(curr_val) >= init_thresh;
                        double diag_el = sol_vec.matr_el_at_pos(det_idx);
                        *sol_vec(next_idx, det_idx) -= eps * diag_el * (curr_val);
                    }
                }
                // curr_idx still contains V
                // next_idx now contains -eps H V
//                for (size_t el_idx = 0; el_idx < htrial_size; el_idx++) {
//                    for (uint16_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
//                        *sol_vec(next_idx, el_idx) -= eps * htrial_vals(trial_idx, el_idx) * trial_iterat(trial_idx, vec_idx);
//                    }
//                }
                
                sol_vec.add_vecs(curr_idx, next_idx);
                sol_vec.set_curr_vec_idx(next_idx);
                sol_vec.zero_vec();
                // project trial vectors out
                for (uint8_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
                    double overlap = 0;
                    for (size_t det_idx = 0; det_idx < trial_size; det_idx++) {
                        overlap += trial_vals(trial_idx, det_idx) * *sol_vec(curr_idx, det_idx);
                    }
                    overlap = sum_mpi(overlap, proc_rank, n_procs);
                    for (size_t det_idx = 0; det_idx < trial_size; det_idx++) {
                        *sol_vec(curr_idx, det_idx) -= overlap * trial_vals(trial_idx, det_idx);
                    }
                }
                
                ini_count = sum_mpi(ini_count, proc_rank, n_procs);
                if (proc_rank == 0) {
                    nini_f << ini_count;
                    if (vec_idx + 1 < n_trial) {
                        nini_f << ',';
                    }
                }
            }
            nini_f << '\n';
            
            for (uint16_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
                for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                    trial_iterat(trial_idx, vec_idx) -= eps * h_mat(trial_idx, vec_idx);
                }
            }
            
            if ((iteration + 1) % save_interval == 0) {
                sol_vec.save(args.result_dir);
                nini_f.flush();
            }
        }
        if (proc_rank == 0) {
            bmat_file.close();
            dmat_file.close();
            shift_f.close();
            norm_f.close();
            nini_f.close();
        }
        MPI_Finalize();
    } catch (std::exception &ex) {
        std::cerr << "\nException : " << ex.what() << "\n\nPlease send a description of this error, a copy of the command-line arguments used, and the random number generator seeds printed for each process to the developers through our GitHub repository: https://github.com/sgreene8/FRIES/ \n\n";
    }
    
    return 0;
}
