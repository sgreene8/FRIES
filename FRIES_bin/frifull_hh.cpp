/*! \file
 *
 * \brief Perform FRI without matrix compression on the Hubbard-Holstein model.
 *
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#define __STDC_LIMIT_MACROS
#include <cstdint>
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
        fprintf(stderr, "Error: target number of nonzero vector elements not specified\n");
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
    unsigned int proc_scrambler[2 * n_orb];
    double loc_norm, glob_norm;
    double last_one_norm = 0;
    
    if (load_dir) {
        load_proc_hash(load_dir, proc_scrambler);
    }
    else {
        if (proc_rank == 0) {
            for (det_idx = 0; det_idx < 2 * n_orb; det_idx++) {
                proc_scrambler[det_idx] = genrand_mt(rngen_ptr);
            }
            save_proc_hash(result_dir, proc_scrambler, 2 * n_orb);
        }
#ifdef USE_MPI
        MPI_Bcast(&proc_scrambler, 2 * n_orb, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
#endif
    }
    
    // Solution vector
    unsigned int spawn_length = n_elec * 4 * max_n_dets / n_procs;
    uint8_t ph_bits = 3;
    HubHolVec<double> sol_vec(max_n_dets, spawn_length, rngen_ptr, hub_len, ph_bits, n_elec, n_procs);
    size_t det_size = CEILING(2 * n_orb + ph_bits * n_orb, 8);
    sol_vec.proc_scrambler_ = proc_scrambler;
    
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
            sol_vec.add(neel_det, 100, 1, 0);
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
    
    int ini_flag;
    double matr_el;
    double recv_nums[n_procs];
    uint8_t new_det[det_size];
    
    uint8_t (*spawn_orbs)[2] = (uint8_t (*)[2])malloc(sizeof(uint8_t) * n_elec * 2 * 2);
    
    unsigned int iterat;
    Matrix<uint8_t> &neighb_orbs = sol_vec.neighb();
    for (iterat = 0; iterat < max_iter; iterat++) {
        for (det_idx = 0; det_idx < sol_vec.curr_size(); det_idx++) {
            double *curr_el = sol_vec[det_idx];
            uint8_t *curr_det = sol_vec.indices()[det_idx];
            ini_flag = fabs(*curr_el) > init_thresh;
            
            size_t n_success = hub_all(n_elec, neighb_orbs[det_idx], spawn_orbs);
            
            for (size_t ex_idx = 0; ex_idx < n_success; ex_idx++) {
                memcpy(new_det, curr_det, det_size);
                zero_bit(new_det, spawn_orbs[ex_idx][0]);
                set_bit(new_det, spawn_orbs[ex_idx][1]);
                sol_vec.add(new_det, eps * hub_t * (*curr_el), ini_flag, 0);
            }
            
            uint8_t *curr_occ = sol_vec.orbs_at_pos(det_idx);
            uint8_t *curr_phonons = sol_vec.phonons_at_pos(det_idx);
            for (size_t elec_idx = 0; elec_idx < n_elec / 2; elec_idx++) {
                uint8_t site = curr_occ[elec_idx];
                uint8_t phonon_num = curr_phonons[site];
                int doubly_occ = read_bit(curr_det, site + hub_len);
                if (phonon_num > 0) {
                    sol_vec.det_from_ph(curr_det, new_det, site, -1);
                    sol_vec.add(new_det, eps * elec_ph * sqrt(phonon_num) * (doubly_occ + 1) * (*curr_el), ini_flag, 0);
                }
                if (phonon_num + 1 < (1 << ph_bits)) {
                    sol_vec.det_from_ph(curr_det, new_det, site, +1);
                    sol_vec.add(new_det, eps * elec_ph * sqrt(phonon_num + 1) * (doubly_occ + 1) * (*curr_el), ini_flag, 0);
                }
            }
            for (size_t elec_idx = n_elec / 2; elec_idx < n_elec; elec_idx++) {
                uint8_t site = curr_occ[elec_idx] - n_orb;
                int doubly_occ = read_bit(curr_det, site);
                if (!doubly_occ) {
                    uint8_t phonon_num = curr_phonons[site];
                    if (phonon_num > 0) {
                        sol_vec.det_from_ph(curr_det, new_det, site, -1);
                        sol_vec.add(new_det, eps * elec_ph * sqrt(phonon_num) * (*curr_el), ini_flag, 0);
                    }
                    if (phonon_num + 1 < (1 << ph_bits)) {
                        sol_vec.det_from_ph(curr_det, new_det, site, +1);
                        sol_vec.add(new_det, eps * elec_ph * sqrt(phonon_num + 1) * (*curr_el), ini_flag, 0);
                    }
                }
            }

            // Death/cloning step
            if (*curr_el != 0) {
                double *diag_el = sol_vec.matr_el_at_pos(det_idx);
                if (isnan(*diag_el)) {
                    *diag_el = hub_diag(curr_det, hub_len, sol_vec.tabl());
                }
                double phonon_diag = sol_vec.total_ph(det_idx) * ph_freq;
                *curr_el *= 1 - eps * (*diag_el * hub_u + phonon_diag - hf_en - en_shift);
            }
        }
        sol_vec.perform_add();
        
        size_t new_max_dets = sol_vec.max_size();
        if (new_max_dets > max_n_dets) {
            keep_exact.resize(new_max_dets, false);
            srt_arr = (size_t *)realloc(srt_arr, sizeof(size_t) * new_max_dets);
            for (; max_n_dets < new_max_dets; max_n_dets++) {
                srt_arr[max_n_dets] = max_n_dets;
            }
        }
        
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
        matr_el = calc_ref_ovlp(sol_vec.indices(), sol_vec.values(), sol_vec.phonon_nums(), sol_vec.curr_size(), neel_det, neel_occ, sol_vec.tabl(), n_elec, hub_len, elec_ph / hub_t);
#ifdef USE_MPI
        MPI_Gather(&matr_el, 1, MPI_DOUBLE, recv_nums, 1, MPI_DOUBLE, ref_proc, MPI_COMM_WORLD);
#else
        recv_nums[0] = matr_el;
#endif
        if (proc_rank == ref_proc) {
            double *diag_el = sol_vec.matr_el_at_pos(0);
            double ref_element = *(sol_vec[0]);
            matr_el = (*diag_el * hub_u - hf_en) * ref_element;
            for (proc_idx = 0; proc_idx < n_procs; proc_idx++) {
                matr_el += recv_nums[proc_idx] * -hub_t;
            }
            fprintf(num_file, "%lf\n", matr_el);
            fprintf(den_file, "%lf\n", ref_element);
            printf("%6u, norm: %lf, en est: %lf, shift: %lf, n_neel: %lf\n", iterat, glob_norm, matr_el / ref_element, en_shift, ref_element);
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

