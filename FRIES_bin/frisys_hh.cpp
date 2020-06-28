/*! \file
 *
 * \brief Perform FRI with systematic matrix compression on the Hubbard-Holstein model
 */

#include <iostream>
#include <ctime>
#include <FRIES/io_utils.hpp>
#include <FRIES/Ext_Libs/dcmt/dc.h>
#include <FRIES/compress_utils.hpp>
#include <FRIES/Ext_Libs/argparse.h>
#include <FRIES/Hamiltonians/hub_holstein.hpp>
#include <FRIES/hh_vec.hpp>

static const char *const usage[] = {
    "frisys_hh [options] [[--] args]",
    "frisys_hh [options]",
    NULL,
};

int main(int argc, const char * argv[]) {
    int n_procs = 1;
    int proc_rank = 0;
    unsigned int proc_idx, ref_proc;
#ifdef USE_MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
#endif
    
    const char *params_path = NULL;
    const char *result_dir = "./";
    const char *load_dir = NULL;
    unsigned int target_nonz = 0;
    unsigned int max_n_dets = 0;
    unsigned int init_thresh = 0;
    unsigned int tmp_norm = 0;
    unsigned int max_iter = 1000000;
    struct argparse_option options[] = {
        OPT_HELP(),
        OPT_STRING('d', "params_path", &params_path, "Path to the file that contains the parameters defining the Hamiltonian, number of electrons, number of sites, etc."),
        OPT_INTEGER('t', "target", &tmp_norm, "Target one-norm of solution vector"),
        OPT_INTEGER('m', "vec_nonz", &target_nonz, "Target number of nonzero vector elements to keep after each iteration"),
        OPT_STRING('y', "result_dir", &result_dir, "Directory in which to save output files"),
        OPT_INTEGER('p', "max_dets", &max_n_dets, "Maximum number of determinants on a single MPI process."),
        OPT_INTEGER('i', "initiator", &init_thresh, "Number of walkers on a determinant required to make it an initiator."),
        OPT_STRING('l', "load_dir", &load_dir, "Directory from which to load checkpoint files from a previous FRI calculation (in binary format, see documentation for DistVec::save() and DistVec::load())."),
        OPT_INTEGER('I', "max_iter", &max_iter, "Maximum number of iterations to run the calculation."),
        OPT_END(),
    };
    
    struct argparse argparse;
    argparse_init(&argparse, options, usage, 0);
    argparse_describe(&argparse, "\nPerform an FCIQMC calculation.", "");
    argc = argparse_parse(&argparse, argc, argv);
    
    if (params_path == NULL) {
        fprintf(stderr, "Error: parameter file not specified.\n");
        return 0;
    }
    if (target_nonz == 0) {
        fprintf(stderr, "Error: target number of nonzero elements in compression not specified\n");
        return 0;
    }
    if (max_n_dets == 0) {
        fprintf(stderr, "Error: maximum number of determinants expected on each processor not specified.\n");
        return 0;
    }
    
    double target_norm = tmp_norm;
    
    // Parameters
    double shift_damping = 0.05;
    unsigned int shift_interval = 10;
    unsigned int save_interval = 1000;
    double en_shift = 0;
    
    // Read in data files
    hh_input in_data;
    parse_hh_input(params_path, &in_data);
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
    mt_struct *rngen_ptr = get_mt_parameter_id_st(32, 521, proc_rank, (unsigned int) time(NULL));
    sgenrand_mt((uint32_t) time(NULL), rngen_ptr);
    size_t det_idx;
    
    // Initialize hash function for processors and vector
    std::vector<uint32_t> proc_scrambler(2 * n_orb);
    double loc_norm, glob_norm;
    double last_one_norm = 0;
    
    if (load_dir) {
        load_proc_hash(load_dir, proc_scrambler.data());
    }
    else {
        if (proc_rank == 0) {
            for (det_idx = 0; det_idx < 2 * n_orb; det_idx++) {
                proc_scrambler[det_idx] = genrand_mt(rngen_ptr);
            }
            save_proc_hash(result_dir, proc_scrambler.data(), 2 * n_orb);
        }
#ifdef USE_MPI
        MPI_Bcast(proc_scrambler.data(), 2 * n_orb, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
#endif
    }
    
    // Solution vector
    unsigned int spawn_length = target_nonz * 4 / n_procs;
    size_t adder_size = spawn_length > 200000 ? 200000 : spawn_length;
    uint8_t ph_bits = 3;
    std::function<double(const uint8_t *)> diag_shortcut = [hub_len](const uint8_t *det) {
        return hub_diag((uint8_t *)det, hub_len);
    };
    HubHolVec<double> sol_vec(max_n_dets, spawn_length, hub_len, ph_bits, n_elec, n_procs, diag_shortcut, 2, proc_scrambler);
    size_t det_size = CEILING(2 * n_orb + ph_bits * n_orb, 8);
    
    uint8_t neel_det[det_size];
    gen_neel_det_1D(n_orb, n_elec, ph_bits, neel_det);
    ref_proc = sol_vec.idx_to_proc(neel_det);
    uint8_t neel_occ[n_elec];
    sol_vec.gen_orb_list(neel_det, neel_occ);
    
    // Initialize solution vector
    if (load_dir) {
        sol_vec.load(load_dir);
    }
    else {
        if (ref_proc == proc_rank) {
            sol_vec.add(neel_det, 100, 1);
        }
    }
    sol_vec.perform_add();
    loc_norm = sol_vec.local_norm();
    glob_norm = sum_mpi(loc_norm, proc_rank, n_procs);
    if (load_dir) {
        last_one_norm = glob_norm;
    }
    
    char file_path[300];
    FILE *norm_file = NULL;
    FILE *num_file = NULL;
    FILE *den_file = NULL;
    FILE *shift_file = NULL;
    if (proc_rank == ref_proc) {
        // Setup output files
        strcpy(file_path, result_dir);
        strcat(file_path, "projnum.txt");
        num_file = fopen(file_path, "a");
        if (!num_file) {
            fprintf(stderr, "Could not open file for writing in directory %s\n", result_dir);
        }
        strcpy(file_path, result_dir);
        strcat(file_path, "projden.txt");
        den_file = fopen(file_path, "a");
        strcpy(file_path, result_dir);
        strcat(file_path, "S.txt");
        shift_file = fopen(file_path, "a");
        strcpy(file_path, result_dir);
        strcat(file_path, "norm.txt");
        norm_file = fopen(file_path, "a");
        
        strcpy(file_path, result_dir);
        strcat(file_path, "params.txt");
        FILE *param_f = fopen(file_path, "w");
        fprintf(param_f, "FRI calculation\nHubbard-Holstein parameters path: %s\nepsilon (imaginary time step): %lf\nTarget norm %lf\nInitiator threshold: %u\nVector nonzero: %u\n", params_path, eps, target_norm, init_thresh, target_nonz);
        if (load_dir) {
            fprintf(param_f, "Restarting calculation from %s\n", load_dir);
        }
        else {
            fprintf(param_f, "Initializing calculation from Neel unit vector\n");
        }
        fclose(param_f);
    }
    
    // Parameters for systematic sampling
    double loc_norms[n_procs];
    double rn_sys = 0;
    size_t *srt_arr = (size_t *)malloc(sizeof(size_t) * max_n_dets);
    for (det_idx = 0; det_idx < max_n_dets; det_idx++) {
        srt_arr[det_idx] = det_idx;
    }
    std::vector<bool> keep_exact(max_n_dets, false);
    
    Matrix<double> subwt_mem(spawn_length, 2);
    Matrix<bool> keep_idx(spawn_length, 2);
    std::vector<double> wt_remain(spawn_length, 0);
    unsigned int *ndiv_vec = (unsigned int *)malloc(sizeof(unsigned int) * spawn_length);
    double *comp_vec1 = (double *)malloc(sizeof(double) * spawn_length);
    double *comp_vec2 = (double *)malloc(sizeof(double) * spawn_length);
    size_t (*comp_idx)[2] = (size_t (*)[2])malloc(sizeof(size_t) * 2 * spawn_length);
    size_t comp_len;
    size_t *det_indices = (size_t *)malloc(sizeof(size_t) * spawn_length);
    std::vector<bool> ph_ex(spawn_length, false);
    uint8_t new_det[det_size];
    double recv_nums[n_procs];
    
    Matrix<uint8_t> &neighb_orbs = sol_vec.neighb();
    for (unsigned int iterat = 0; iterat < max_iter; iterat++) {
        for (det_idx = 0; det_idx < sol_vec.curr_size(); det_idx++) {
            double *curr_el = sol_vec[det_idx];
            double weight = fabs(*curr_el);
            comp_vec1[det_idx] = weight;
            if (weight > 0) {
                subwt_mem(det_idx, 0) = hub_t;
                subwt_mem(det_idx, 1) = elec_ph;
                ndiv_vec[det_idx] = 0;
            }
            else {
                ndiv_vec[det_idx] = 1;
            }
        }
        if (proc_rank == 0) {
            rn_sys = genrand_mt(rngen_ptr) / (1. + UINT32_MAX);
        }
        comp_len = comp_sub(comp_vec1, sol_vec.curr_size(), ndiv_vec, subwt_mem, keep_idx, NULL, target_nonz, wt_remain.data(), rn_sys, comp_vec2, comp_idx);
        if (comp_len > spawn_length) {
            fprintf(stderr, "Error: insufficient memory allocated for matrix compression.\n");
        }
        
        for (size_t samp_idx = 0; samp_idx < comp_len; samp_idx++) {
            det_idx = comp_idx[samp_idx][0];
            det_indices[samp_idx] = det_idx;
            ph_ex[samp_idx] = comp_idx[samp_idx][1];
            if (ph_ex[samp_idx]) { // phonon excitation
                ndiv_vec[samp_idx] = 2 * n_elec;
            }
            else { // electron excitation
                ndiv_vec[samp_idx] = neighb_orbs(det_idx, 0) + neighb_orbs(det_idx, n_elec + 1);
            }
            comp_vec2[samp_idx] *= ndiv_vec[samp_idx];
        }
        if (proc_rank == 0) {
            rn_sys = genrand_mt(rngen_ptr) / (1. + UINT32_MAX);
        }
        comp_len = comp_sub(comp_vec2, comp_len, ndiv_vec, subwt_mem, keep_idx, NULL, target_nonz, wt_remain.data(), rn_sys, comp_vec1, comp_idx);
        if (comp_len > spawn_length) {
            fprintf(stderr, "Error: insufficient memory allocated for matrix compression.\n");
        }
        
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
                    size_t prev_idx = comp_idx[samp_idx][0];
                    size_t det_idx = det_indices[prev_idx];
                    double curr_val = vals_before_mult[det_idx];
                    uint8_t ini_flag = fabs(curr_val) >= init_thresh;
                    if (ini_flag != add_ini) {
                        samp_idx++;
                        continue;
                    }
                    uint8_t exc_idx = comp_idx[samp_idx][1];
                    uint8_t *curr_det = sol_vec.indices()[det_idx];
                    double matr_el = comp_vec1[samp_idx] * -eps;
                    if (curr_val < 0) {
                        matr_el *= -1;
                    }
                    if (ph_ex[prev_idx]) {
                        uint8_t *curr_ph = sol_vec.phonons_at_pos(det_idx);
                        uint8_t *curr_occ = sol_vec.orbs_at_pos(det_idx);
                        uint8_t site = curr_occ[exc_idx % n_elec] % hub_len;
                        uint8_t phonon_num = curr_ph[site];
                        if (exc_idx < n_elec && phonon_num > 0) {
                            sol_vec.det_from_ph(curr_det, new_det, site, -1);
                            matr_el *= sqrt(phonon_num);
                        }
                        else if (exc_idx >= n_elec && phonon_num + 1 < (1 << ph_bits)) {
                            sol_vec.det_from_ph(curr_det, new_det, site, +1);
                            matr_el *= sqrt(phonon_num + 1);
                        }
                        else {
                            matr_el = 0;
                        }
                    }
                    else {
                        std::copy(curr_det, curr_det + det_size, new_det);
                        uint8_t *curr_neighb = neighb_orbs[det_idx];
                        uint8_t orig_orb;
                        uint8_t dest_orb;
                        if (exc_idx < curr_neighb[0]) {
                            orig_orb = curr_neighb[exc_idx + 1];
                            dest_orb = orig_orb + 1;
                        }
                        else {
                            orig_orb = curr_neighb[n_elec + 1 + exc_idx - curr_neighb[0] + 1];
                            dest_orb = orig_orb - 1;
                        }
                        zero_bit(new_det, orig_orb);
                        set_bit(new_det, dest_orb);
                        matr_el *= -1; // hub_t
                    }
                    if (fabs(matr_el) > 1e-9) {
                        sol_vec.add(new_det, matr_el, ini_flag);
                        num_added++;
                    }
                    samp_idx++;
                }
                sol_vec.perform_add();
                num_added = sum_mpi(num_added, proc_rank, n_procs);
            }
        }
        size_t new_max_dets = sol_vec.max_size();
        if (new_max_dets > max_n_dets) {
            keep_exact.resize(new_max_dets, false);
            srt_arr = (size_t *)realloc(srt_arr, sizeof(size_t) * new_max_dets);
            for (; max_n_dets < new_max_dets; max_n_dets++) {
                srt_arr[max_n_dets] = max_n_dets;
            }
        }
        
#pragma mark Diagonal multiplication
        sol_vec.set_curr_vec_idx(0);
        for (det_idx = 0; det_idx < vec_size; det_idx++) {
            double *curr_el = sol_vec[det_idx];
            if (*curr_el != 0) {
                double diag_el = sol_vec.matr_el_at_pos(det_idx);
                double phonon_diag = sol_vec.total_ph(det_idx) * ph_freq;
                *curr_el *= 1 - eps * (diag_el * hub_u + phonon_diag - hf_en - en_shift);
            }
        }
        sol_vec.add_vecs(0, 1);
        
        // Compression step
        unsigned int n_samp = target_nonz;
        loc_norms[proc_rank] = find_preserve(sol_vec.values(), srt_arr, keep_exact, sol_vec.curr_size(), &n_samp, &glob_norm);
        
        // Adjust shift
        if ((iterat + 1) % shift_interval == 0) {
            adjust_shift(&en_shift, glob_norm, &last_one_norm, target_norm, shift_damping / shift_interval / eps);
            if (proc_rank == ref_proc) {
                fprintf(shift_file, "%lf\n", en_shift);
                fprintf(norm_file, "%lf\n", glob_norm);
            }
        }
        
        // Calculate energy estimate
        double numer = calc_ref_ovlp(sol_vec.indices(), sol_vec.values(), sol_vec.phonon_nums(), sol_vec.curr_size(), neel_det, neel_occ, n_elec, hub_len, elec_ph / hub_t);
#ifdef USE_MPI
        MPI_Gather(&numer, 1, MPI_DOUBLE, recv_nums, 1, MPI_DOUBLE, ref_proc, MPI_COMM_WORLD);
#else
        recv_nums[0] = numer;
#endif
        if (proc_rank == ref_proc) {
            double diag_el = sol_vec.matr_el_at_pos(0);
            double ref_element = *(sol_vec[0]);
            numer = (diag_el * hub_u - hf_en) * ref_element;
            for (proc_idx = 0; proc_idx < n_procs; proc_idx++) {
                numer += recv_nums[proc_idx] * -hub_t;
            }
            fprintf(num_file, "%lf\n", numer);
            fprintf(den_file, "%lf\n", ref_element);
            printf("%6u, norm: %lf, en est: %lf, shift: %lf, n_neel: %lf\n", iterat, glob_norm, numer / ref_element, en_shift, ref_element);
        }
        
        if (proc_rank == 0) {
            rn_sys = genrand_mt(rngen_ptr) / (1. + UINT32_MAX);
        }
#ifdef USE_MPI
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, loc_norms, 1, MPI_DOUBLE, MPI_COMM_WORLD);
#endif
        sys_comp(sol_vec.values(), sol_vec.curr_size(), loc_norms, n_samp, keep_exact, rn_sys);
        for (det_idx = 0; det_idx < sol_vec.curr_size(); det_idx++) {
            if (keep_exact[det_idx] && !(proc_rank == 0 && det_idx == 0)) {
                sol_vec.del_at_pos(det_idx);
                keep_exact[det_idx] = 0;
            }
        }
        
        // Save vector snapshot to disk
        if ((iterat + 1) % save_interval == 0) {
            sol_vec.save(result_dir);
            if (proc_rank == ref_proc) {
                fflush(num_file);
                fflush(den_file);
                fflush(shift_file);
            }
        }
    }
        sol_vec.save(result_dir);
        if (proc_rank == ref_proc) {
            fclose(num_file);
            fclose(den_file);
            fclose(shift_file);
        }
    #ifdef USE_MPI
        MPI_Finalize();
    #endif
    return 0;
}
