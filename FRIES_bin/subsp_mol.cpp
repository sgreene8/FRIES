/*! \file
 *
 * \brief FRI with unnormalized HB-PP matrix factorization applied to subspace iteration for calculating excited-state energies.
 */

#include <iostream>
#include <chrono>
#include <FRIES/io_utils.hpp>
#include <FRIES/compress_utils.hpp>
#include <FRIES/Ext_Libs/argparse.hpp>
#include <FRIES/Hamiltonians/near_uniform.hpp>
#include <FRIES/Hamiltonians/heat_bathPP.hpp>
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
    double init_thresh = kwarg("initiator", "Magnitude of vector element required to make it an initiator").set_default(0);
    std::shared_ptr<std::string> load_dir = kwarg("load_dir", "Directory from which to load checkpoint files from a previous FRI calculation (in binary format, see documentation for DistVec::save() and DistVec::load())");
    
    CONSTRUCTOR(MyArgs);
};

int main(int argc, char * argv[]) {
    MyArgs args(argc, argv);
    
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
        double shift_damping = 0.05;
        unsigned int shift_interval = 10;
        unsigned int save_interval = 100;
        double en_shift = 0;

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
        
        unsigned int spawn_length = matr_samp * 4 / n_procs;
        size_t adder_size = spawn_length > 1000000 ? 1000000 : spawn_length;
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
        
        Adder<double> shared_adder(adder_size, n_procs, n_orb * 2);
        DistVec<double> sol_vec(args.max_n_dets, &shared_adder, n_orb * 2, n_elec_unf, diag_shortcut, 2 * n_trial, proc_scrambler, vec_scrambler);
        
        uint8_t hf_det[det_size];
        gen_hf_bitstring(n_orb, n_elec - n_frz, hf_det);
        hf_proc = sol_vec.idx_to_proc(hf_det);
        
        uint8_t tmp_orbs[n_elec_unf];
        uint8_t (*orb_indices1)[4] = (uint8_t (*)[4])malloc(sizeof(char) * 4 * spawn_length);

        # pragma mark Set up trial vectors
        std::vector<DistVec<double>> trial_vecs;
        trial_vecs.reserve(n_trial);
        std::vector<DistVec<double>> htrial_vecs;
        htrial_vecs.reserve(n_trial);
        
        size_t n_ex = n_orb * n_orb * n_elec_unf * n_elec_unf;
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
            htrial_vecs.emplace_back(glob_n_dets * n_ex / n_procs, &shared_adder, n_orb * 2, n_elec_unf, diag_shortcut, 2, proc_scrambler, vec_scrambler);
            
            for (size_t det_idx = 0; det_idx < loc_n_dets; det_idx++) {
                trial_vecs[trial_idx].add((*load_dets)[det_idx], load_vals[det_idx], 1);
                htrial_vecs[trial_idx].add((*load_dets)[det_idx], load_vals[det_idx], 1);
            }
            trial_vecs[trial_idx].perform_add();
            htrial_vecs[trial_idx].perform_add();
            
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
        std::vector<std::vector<uintmax_t>> htrial_hashes(n_trial);
        for (uint8_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
            uint8_t tmp_orbs[n_elec_unf];
            DistVec<double> &curr_trial = trial_vecs[trial_idx];
            curr_trial.collect_procs();
            trial_hashes[trial_idx].resize(curr_trial.curr_size());
            for (size_t det_idx = 0; det_idx < curr_trial.curr_size(); det_idx++) {
                trial_hashes[trial_idx][det_idx] = sol_vec.idx_to_hash(curr_trial.indices()[det_idx], tmp_orbs);
            }
            
            DistVec<double> &curr_htrial = htrial_vecs[trial_idx];
            h_op_offdiag(curr_htrial, symm, tot_orb, *eris, *h_core, (uint8_t *)orb_indices1, n_frz, n_elec_unf, 1, 1);
            curr_htrial.set_curr_vec_idx(0);
            h_op_diag(curr_htrial, 0, 0, 1);
            curr_htrial.collect_procs();
            
            htrial_hashes[trial_idx].resize(curr_htrial.curr_size());
            for (det_idx = 0; det_idx < curr_htrial.curr_size(); det_idx++) {
                htrial_hashes[trial_idx][det_idx] = sol_vec.idx_to_hash(curr_htrial.indices()[det_idx], tmp_orbs);
            }
        }
        
        // Count # single/double excitations from HF
        sol_vec.gen_orb_list(hf_det, tmp_orbs);
        size_t n_hf_doub = doub_ex_symm(hf_det, tmp_orbs, n_elec_unf, n_orb, orb_indices1, symm);
        size_t n_hf_sing = count_singex(hf_det, tmp_orbs, symm, n_orb, symm_lookup, n_elec_unf);
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
        
        size_t n_states = n_elec_unf > (n_orb - n_elec_unf / 2) ? n_elec_unf : n_orb - n_elec_unf / 2;
        Matrix<double> subwt_mem(spawn_length, n_states);
        uint16_t *sub_sizes = (uint16_t *)malloc(sizeof(uint16_t) * spawn_length);
        unsigned int *ndiv_vec = (unsigned int *)malloc(sizeof(unsigned int) * spawn_length);
        double *comp_vec1 = (double *)malloc(sizeof(double) * spawn_length);
        double *comp_vec2 = (double *)malloc(sizeof(double) * spawn_length);
        size_t (*comp_idx)[2] = (size_t (*)[2])malloc(sizeof(size_t) * 2 * spawn_length);
        size_t comp_len;
        size_t *det_indices1 = (size_t *)malloc(sizeof(size_t) * 2 * spawn_length);
        size_t *det_indices2 = &det_indices1[spawn_length];
        uint8_t (*orb_indices2)[4] = (uint8_t (*)[4])malloc(sizeof(uint8_t) * 4 * spawn_length);
        unsigned int unocc_symm_cts[n_irreps][2];
        Matrix<bool> keep_idx(spawn_length, n_states);
        double *wt_remain = (double *)calloc(spawn_length, sizeof(double));
        size_t samp_idx, weight_idx;
        
        hb_info *hb_probs = set_up(tot_orb, n_orb, *eris);
        
        // Vectors for systematic sampling
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
                    std::fill(en_shifts.begin(), en_shifts.end(), 1);
            }
            if ((iteration + 1) % shift_interval == 0) {
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
            for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                sol_vec.set_curr_vec_idx(vec_half * n_trial + vec_idx);
                for (size_t el_idx = 0; el_idx < sol_vec.curr_size(); el_idx++) {
                    *sol_vec[el_idx] /= en_shifts[vec_idx];
                }
            }
                        
#pragma mark Calculate overlap and hamiltonian matrices
            for (uint16_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
                DistVec<double> &curr_trial = trial_vecs[trial_idx];
                for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                    sol_vec.set_curr_vec_idx(vec_half * n_trial + vec_idx);
                    double d_prod = sol_vec.dot(curr_trial.indices(), curr_trial.values(), curr_trial.curr_size(), trial_hashes[trial_idx]);
                    d_mat(trial_idx, vec_idx) = sum_mpi(d_prod, proc_rank, n_procs, MPI_COMM_WORLD);
                }
            }
            for (uint16_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
                DistVec<double> &curr_htrial = htrial_vecs[trial_idx];
                for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                    sol_vec.set_curr_vec_idx(vec_half * n_trial + vec_idx);
                    double d_prod = sol_vec.dot(curr_htrial.indices(), curr_htrial.values(), curr_htrial.curr_size(), htrial_hashes[trial_idx]);
                    h_mat(trial_idx, vec_idx) = sum_mpi(d_prod, proc_rank, n_procs, MPI_COMM_WORLD);
                }
            }

#pragma mark Vector compression
            compress_vecs(sol_vec, vec_half * n_trial, (vec_half + 1) * n_trial, args.target_nonz, srt_arr, keep_exact, del_all, mt_obj);

# pragma mark Matrix multiplication
            for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                int curr_idx = vec_half * n_trial + vec_idx;
                int next_idx = (1 - vec_half) * n_trial + vec_idx;
                sol_vec.set_curr_vec_idx(curr_idx);

#pragma mark Singles vs doubles
                subwt_mem.reshape(spawn_length, 2);
                keep_idx.reshape(spawn_length, 2);
                for (det_idx = 0; det_idx < sol_vec.curr_size(); det_idx++) {
                    double *curr_el = sol_vec[det_idx];
                    double weight = fabs(*curr_el);
                    comp_vec1[det_idx] = weight;
                    if (weight > 0) {
                        subwt_mem(det_idx, 0) = p_doub;
                        subwt_mem(det_idx, 1) = (1 - p_doub);
                        ndiv_vec[det_idx] = 0;
                    }
                    else {
                        ndiv_vec[det_idx] = 1;
                    }
                }
                double rn_sys = 0;
                if (proc_rank == 0) {
                    rn_sys = mt_obj() / (1. + UINT32_MAX);
                }
                comp_len = comp_sub(comp_vec1, sol_vec.curr_size(), ndiv_vec, subwt_mem, keep_idx, NULL, matr_samp, wt_remain, rn_sys, comp_vec2, comp_idx);
                if (comp_len > spawn_length) {
                    std::cerr << "Error: insufficient memory allocated for matrix compression.\n";
                }

#pragma mark  First occupied orbital
                subwt_mem.reshape(spawn_length, n_elec_unf - 1);
                keep_idx.reshape(spawn_length, n_elec_unf - 1);
                for (samp_idx = 0; samp_idx < comp_len; samp_idx++) {
                    det_idx = comp_idx[samp_idx][0];
                    det_indices1[samp_idx] = det_idx;
                    orb_indices1[samp_idx][0] = comp_idx[samp_idx][1];
                    uint8_t *occ_orbs = sol_vec.orbs_at_pos(det_idx);
                    if (orb_indices1[samp_idx][0] == 0) { // double excitation
                        ndiv_vec[samp_idx] = 0;
                        double tot_weight = calc_o1_probs(hb_probs, subwt_mem[samp_idx], n_elec_unf, occ_orbs, true);
                        comp_vec2[samp_idx] *= tot_weight;
                    }
                    else {
                        count_symm_virt(unocc_symm_cts, occ_orbs, n_elec_unf, n_orb, n_irreps, symm_lookup, symm);
                        unsigned int n_occ = count_sing_allowed(occ_orbs, n_elec_unf, symm, n_orb, unocc_symm_cts);
                        if (n_occ == 0) {
                            ndiv_vec[samp_idx] = 1;
                            comp_vec2[samp_idx] = 0;
                        }
                        else {
                            ndiv_vec[samp_idx] = n_occ;
                        }
                    }
                }
                
                if (proc_rank == 0) {
                    rn_sys = mt_obj() / (1. + UINT32_MAX);
                }
                comp_len = comp_sub(comp_vec2, comp_len, ndiv_vec, subwt_mem, keep_idx, NULL, matr_samp, wt_remain, rn_sys, comp_vec1, comp_idx);
                if (comp_len > spawn_length) {
                    std::cerr << "Error: insufficient memory allocated for matrix compression.\n";
                }
                            
#pragma mark Unoccupied orbital (single); 2nd occupied (double)
                for (samp_idx = 0; samp_idx < comp_len; samp_idx++) {
                    weight_idx = comp_idx[samp_idx][0];
                    det_idx = det_indices1[weight_idx];
                    det_indices2[samp_idx] = det_idx;
                    orb_indices2[samp_idx][0] = orb_indices1[weight_idx][0]; // single or double
                    orb_indices2[samp_idx][1] = comp_idx[samp_idx][1]; // first occupied orbital index (NOT converted to orbital below)
                    if (orb_indices2[samp_idx][1] >= n_elec_unf) {
                        std::cerr << "Error: chosen occupied orbital (first) is out of bounds\n";
                        comp_vec1[samp_idx] = 0;
                        ndiv_vec[samp_idx] = 1;
                        continue;
                    }
                    uint8_t *occ_orbs = sol_vec.orbs_at_pos(det_idx);
                    if (orb_indices2[samp_idx][0] == 0) { // double excitation
                        ndiv_vec[samp_idx] = 0;
                        if (qmc_dist == heat_bath) {
                            calc_o2_probs(hb_probs, subwt_mem[samp_idx], n_elec_unf, occ_orbs, orb_indices2[samp_idx][1]);
                        }
                        else {
                            orb_indices2[samp_idx][1]++;
                            sub_sizes[samp_idx] = orb_indices2[samp_idx][1];
                            comp_vec1[samp_idx] *= calc_o2_probs_half(hb_probs, subwt_mem[samp_idx], n_elec_unf, occ_orbs, orb_indices2[samp_idx][1]);
                        }
                    }
                    else { // single excitation
                        count_symm_virt(unocc_symm_cts, occ_orbs, n_elec_unf, n_orb, n_irreps, symm_lookup, symm);
                        unsigned int n_virt = count_sing_virt(occ_orbs, n_elec_unf, symm, n_orb, unocc_symm_cts, &orb_indices2[samp_idx][1]);
                        if (n_virt == 0) {
                            ndiv_vec[samp_idx] = 1;
                            comp_vec1[samp_idx] = 0;
                        }
                        else {
                            ndiv_vec[samp_idx] = n_virt;
                            orb_indices2[samp_idx][3] = n_virt; // number of allowed virtual orbitals
                        }
                    }
                }
                if (proc_rank == 0) {
                    rn_sys = mt_obj() / (1. + UINT32_MAX);
                }
                comp_len = comp_sub(comp_vec1, comp_len, ndiv_vec, subwt_mem, keep_idx, new_hb ? sub_sizes : NULL, matr_samp - tot_dense_h, wt_remain, rn_sys, comp_vec2, comp_idx);
                if (comp_len > spawn_length) {
                    std::cerr << "Error: insufficient memory allocated for matrix compression.\n";
                }
            }
            
        }
        
        MPI_Finalize();
    } catch (std::exception &ex) {
        std::cerr << "\nException : " << ex.what() << "\n\nPlease send a description of this error, a copy of the command-line arguments used, and the random number generator seeds printed for each process to the developers through our GitHub repository: https://github.com/sgreene8/FRIES/ \n\n";
    }
    
    return 0;
}
