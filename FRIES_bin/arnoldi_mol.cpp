/*! \file
 *
 * \brief FRI applied to Arnoldi method for calculating excited-state energies
 */

#include <cstdio>
#include <iostream>
#include <ctime>
#include <FRIES/io_utils.hpp>
#include <FRIES/Ext_Libs/dcmt/dc.h>
#include <FRIES/compress_utils.hpp>
#include <FRIES/Ext_Libs/argparse.h>
#include <FRIES/Hamiltonians/heat_bathPP.hpp>
#include <FRIES/Hamiltonians/molecule.hpp>
#include <FRIES/Hamiltonians/near_uniform.hpp>
#include <FRIES/Ext_Libs/LAPACK/lapacke.h>

static const char *const usage[] = {
    "arnoldi_mol [options] [[--] args]",
    "arnoldi_mol [options]",
    NULL,
};


int main(int argc, const char * argv[]) {
    const char *hf_path = NULL;
    const char *result_dir = "./";
    const char *trial_path = NULL;
    unsigned int n_trial = 0;
    unsigned int target_nonz = 0;
    unsigned int matr_samp = 0;
    unsigned int max_n_dets = 0;
    unsigned int max_iter = 500000;
    unsigned int n_krylov = 500;
    struct argparse_option options[] = {
        OPT_HELP(),
        OPT_STRING('d', "hf_path", &hf_path, "Path to the directory that contains the HF output files eris.txt, hcore.txt, symm.txt, hf_en.txt, and sys_params.txt"),
        OPT_INTEGER('m', "vec_nonz", &target_nonz, "Target number of nonzero vector elements to keep after each iteration"),
        OPT_INTEGER('M', "mat_nonz", &matr_samp, "Target number of nonzero matrix elements to keep after each iteration"),
        OPT_STRING('y', "result_dir", &result_dir, "Directory in which to save output files"),
        OPT_INTEGER('p', "max_dets", &max_n_dets, "Maximum number of determinants on a single MPI process."),
        OPT_STRING('v', "trial_vecs", &trial_path, "Prefix for files containing the vectors with which to calculate the energy and initialize the calculation (unless load_dir is provided). Files must have names <trial_vecs>dets<xx> and <trial_vecs>vals<xx>, where xx is a 2-digit number ranging from 0 to (n_trial - 1), and be text files."),
        OPT_INTEGER('k', "num_trial", &n_trial, "Number of trial vectors to use to calculate dot products with the iterates."),
        OPT_INTEGER('I', "max_iter", &max_iter, "Maximum number of iterations to run the calculation."),
        OPT_END(),
    };
    
    struct argparse argparse;
    argparse_init(&argparse, options, usage, 0);
    argparse_describe(&argparse, "\nRandomized Arnoldi method for calculating excited states", "");
    argc = argparse_parse(&argparse, argc, argv);
    
    if (hf_path == NULL) {
        fprintf(stderr, "Error: HF directory not specified.\n");
        return 0;
    }
    if (target_nonz == 0) {
        fprintf(stderr, "Error: target number of nonzero vector elements not specified\n");
        return 0;
    }
    if (matr_samp == 0) {
        fprintf(stderr, "Error: target number of nonzero matrix elements not specified\n");
        return 0;
    }
    if (max_n_dets == 0) {
        fprintf(stderr, "Error: maximum number of determinants expected on each processor not specified.\n");
        return 0;
    }
    if (!trial_path) {
        fprintf(stderr, "Error: path to trial vectors was not specified.\n");
        return 0;
    }
    if (n_trial < 2) {
        fprintf(stderr, "Error: you are either using 1 or 0 trial vectors. Consider using the power method instead of Arnoldi in this case.\n");
        return 0;
    }
    
    int n_procs = 1;
    int proc_rank = 0;
#ifdef USE_MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
#endif
    
    // Read in data files
    hf_input in_data;
    parse_hf_input(hf_path, &in_data);
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
    mt_struct *rngen_ptr = get_mt_parameter_id_st(32, 521, proc_rank, (unsigned int) time(NULL));
    sgenrand_mt((uint32_t) time(NULL), rngen_ptr);
    
    // Solution vector
    unsigned int spawn_length = matr_samp * 4 / n_procs;
    size_t adder_size = spawn_length > 1000000 ? 1000000 : spawn_length;
    std::function<double(const uint8_t *)> diag_shortcut = [tot_orb, eris, h_core, n_frz, n_elec, hf_en](const uint8_t *occ_orbs) {
        return diag_matrel(occ_orbs, tot_orb, *eris, *h_core, n_frz, n_elec) - hf_en;
    };

    // Initialize hash function for processors and vector
    std::vector<uint32_t> proc_scrambler(2 * n_orb);
    
    if (proc_rank == 0) {
        for (size_t det_idx = 0; det_idx < 2 * n_orb; det_idx++) {
            proc_scrambler[det_idx] = genrand_mt(rngen_ptr);
        }
        save_proc_hash(result_dir, proc_scrambler.data(), 2 * n_orb);
    }
#ifdef USE_MPI
    MPI_Bcast(proc_scrambler.data(), 2 * n_orb, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
#endif

    std::vector<uint32_t> vec_scrambler(2 * n_orb);
    for (size_t det_idx = 0; det_idx < 2 * n_orb; det_idx++) {
        vec_scrambler[det_idx] = genrand_mt(rngen_ptr);
    }

    std::vector<DistVec<double>> sol_vecs;
    sol_vecs.reserve(n_trial);
    for (uint8_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
        sol_vecs.emplace_back(max_n_dets, adder_size, n_orb * 2, n_elec_unf, n_procs, diag_shortcut, nullptr, 3, proc_scrambler, vec_scrambler);
    }
    size_t det_size = CEILING(2 * n_orb, 8);
    
    Matrix<uint8_t> symm_lookup(n_irreps, n_orb + 1);
    gen_symm_lookup(symm, symm_lookup);
    unsigned int max_n_symm = 0;
    for (uint8_t irrep_idx = 0; irrep_idx < n_irreps; irrep_idx++) {
        if (symm_lookup[irrep_idx][0] > max_n_symm) {
            max_n_symm = symm_lookup[irrep_idx][0];
        }
    }
    
    uint8_t tmp_orbs[n_elec_unf];
    uint8_t (*orb_indices1)[4] = (uint8_t (*)[4])malloc(sizeof(char) * 4 * spawn_length);
    
# pragma mark Set up trial vectors
    std::vector<DistVec<double>> trial_vecs;
    trial_vecs.reserve(n_trial);
    std::vector<DistVec<double>> htrial_vecs;
    htrial_vecs.reserve(n_trial);
    size_t n_ex = n_orb * n_orb * n_elec_unf * n_elec_unf;
    
    char vec_path[300];
    Matrix<uint8_t> *load_dets = new Matrix<uint8_t>(max_n_dets, det_size);
    for (unsigned int trial_idx = 0; trial_idx < n_trial; trial_idx++) {
        DistVec<double> &curr_sol = sol_vecs[trial_idx];
        double *load_vals = curr_sol.values();
        
        sprintf(vec_path, "%s%02d", trial_path, trial_idx);
        unsigned int loc_n_dets = (unsigned int) load_vec_txt(vec_path, *load_dets, load_vals, DOUB);
        size_t glob_n_dets = loc_n_dets;
#ifdef USE_MPI
        MPI_Bcast(&glob_n_dets, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
#endif
        trial_vecs.emplace_back(glob_n_dets, glob_n_dets, n_orb * 2, n_elec_unf, n_procs, proc_scrambler, vec_scrambler);
        htrial_vecs.emplace_back(glob_n_dets * n_ex / n_procs, glob_n_dets * n_ex / n_procs, n_orb * 2, n_elec_unf, n_procs, diag_shortcut, (double *)NULL, 2, proc_scrambler, vec_scrambler);
        
        sol_vecs[trial_idx].set_curr_vec_idx(2);
        for (size_t det_idx = 0; det_idx < loc_n_dets; det_idx++) {
            trial_vecs[trial_idx].add(load_dets[0][det_idx], load_vals[det_idx], 1);
            htrial_vecs[trial_idx].add(load_dets[0][det_idx], load_vals[det_idx], 1);
            sol_vecs[trial_idx].add(load_dets[0][det_idx], load_vals[det_idx], 1);
        }
        loc_n_dets++; // just to be safe
        bzero(load_vals, loc_n_dets * sizeof(double));
        sol_vecs[trial_idx].perform_add();
    }
    delete load_dets;
    
    std::vector<std::vector<uintmax_t>> trial_hashes(n_trial);
    std::vector<std::vector<uintmax_t>> htrial_hashes(n_trial);
    for (uint8_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
        DistVec<double> &curr_trial = trial_vecs[trial_idx];
        curr_trial.perform_add();
        curr_trial.collect_procs();
        trial_hashes[trial_idx].reserve(curr_trial.curr_size());
        for (size_t det_idx = 0; det_idx < curr_trial.curr_size(); det_idx++) {
            trial_hashes[trial_idx][det_idx] = sol_vecs[0].idx_to_hash(curr_trial.indices()[det_idx], tmp_orbs);
        }
        
        DistVec<double> &curr_htrial = htrial_vecs[trial_idx];
        curr_htrial.perform_add();
        h_op_offdiag(curr_htrial, symm, tot_orb, *eris, *h_core, (uint8_t *)orb_indices1, n_frz, n_elec_unf, 1, 1);
        curr_htrial.set_curr_vec_idx(0);
        h_op_diag(curr_htrial, 0, 0, 1);
        curr_htrial.add_vecs(0, 1);
        curr_htrial.collect_procs();
        htrial_hashes[trial_idx].reserve(   curr_htrial.curr_size());
        for (size_t det_idx = 0; det_idx < curr_htrial.curr_size(); det_idx++) {
            htrial_hashes[trial_idx][det_idx] = sol_vecs[0].idx_to_hash(curr_htrial.indices()[det_idx], tmp_orbs);
        }
    }
    
    char file_path[300];
    FILE *bmat_file = NULL;
    FILE *dmat_file = NULL;
    
    if (proc_rank == 0) {
        // Setup output files
        strcpy(file_path, result_dir);
        strcat(file_path, "b_matrix.txt");
        bmat_file = fopen(file_path, "a");
        if (!bmat_file) {
            fprintf(stderr, "Could not open file for writing in directory %s\n", result_dir);
        }
        
        strcpy(file_path, result_dir);
        strcat(file_path, "d_matrix.txt");
        dmat_file = fopen(file_path, "a");
        
        strcpy(file_path, result_dir);
        strcat(file_path, "params.txt");
        FILE *param_f = fopen(file_path, "w");
        fprintf(param_f, "Arnoldi calculation\nHF path: %s\nepsilon (imaginary time step): %lf\nMatrix nonzero: %u\nVector nonzero: %u\n", hf_path, eps, matr_samp, target_nonz);
        fprintf(param_f, "Path for trial vectors: %s\n", trial_path);
        fprintf(param_f, "Krylov iterations: %u\n", n_krylov);
        fclose(param_f);
    }
    
//    size_t n_states = n_elec_unf > (n_orb - n_elec_unf / 2) ? n_elec_unf : n_orb - n_elec_unf / 2;
//    Matrix<double> subwt_mem(spawn_length, n_states);
//    uint16_t *sub_sizes = (uint16_t *)malloc(sizeof(uint16_t) * spawn_length);
//    unsigned int *ndiv_vec = (unsigned int *)malloc(sizeof(unsigned int) * spawn_length);
//    double *comp_vec1 = (double *)malloc(sizeof(double) * spawn_length);
//    double *comp_vec2 = (double *)malloc(sizeof(double) * spawn_length);
//    size_t (*comp_idx)[2] = (size_t (*)[2])malloc(sizeof(size_t) * 2 * spawn_length);
//    size_t comp_len;
//    size_t *det_indices1 = (size_t *)malloc(sizeof(size_t) * 2 * spawn_length);
//    size_t *det_indices2 = &det_indices1[spawn_length];
//    uint8_t (*orb_indices2)[4] = (uint8_t (*)[4])malloc(sizeof(uint8_t) * 4 * spawn_length);
//    unsigned int unocc_symm_cts[n_irreps][2];
//    Matrix<bool> keep_idx(spawn_length, n_states);
//    double *wt_remain = (double *)calloc(spawn_length, sizeof(double));
//    size_t samp_idx, weight_idx;
    
//    hb_info *hb_probs = set_up(tot_orb, n_orb, *eris);
    
    // Parameters for systematic sampling
    double rn_sys = 0;
    double loc_norms[n_procs];
    for (unsigned int vec_idx = 0; vec_idx < n_trial; vec_idx++) {
        if (sol_vecs[vec_idx].max_size() > max_n_dets) {
            max_n_dets = (unsigned int)sol_vecs[vec_idx].max_size();
        }
    }
    std::vector<size_t> srt_arr(max_n_dets);
    for (size_t det_idx = 0; det_idx < max_n_dets; det_idx++) {
        srt_arr[det_idx] = det_idx;
    }
    std::vector<bool> keep_exact(max_n_dets, false);
    
    for (uint32_t iteration = 0; iteration < max_iter; iteration++) {
        // Initialize the solution vectors
        for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
            sol_vecs[vec_idx].copy_vec(2, 0);
        }
        if (proc_rank == 0) {
            printf("Macro iteration %u\n", iteration);
        }
        
        for (uint16_t krylov_idx = 0; krylov_idx < n_krylov; krylov_idx++) {
#pragma mark Krylov dot products
            for (uint16_t trial_idx = 0; trial_idx < n_trial; trial_idx++) {
                DistVec<double> &curr_trial = trial_vecs[trial_idx];
                DistVec<double> &curr_htrial = htrial_vecs[trial_idx];
                for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                    double d_prod = sol_vecs[vec_idx].dot(curr_trial.indices(), curr_trial.values(), curr_trial.curr_size(), trial_hashes[trial_idx].data());
                    d_prod = sum_mpi(d_prod, proc_rank, n_procs);
                    if (proc_rank == 0) {
                        fprintf(dmat_file, "%lf,", d_prod);
                    }
                    
                    d_prod = sol_vecs[vec_idx].dot(curr_htrial.indices(), curr_htrial.values(), curr_htrial.curr_size(), htrial_hashes[trial_idx].data());
                    d_prod = sum_mpi(d_prod, proc_rank, n_procs);
                    if (proc_rank == 0) {
                        fprintf(bmat_file, "%lf,", d_prod);
                    }
                }
                if (proc_rank == 0) {
                    fprintf(dmat_file, "\n");
                    fprintf(bmat_file, "\n");
                }
            }
            if (proc_rank == 0) {
                printf("Krylov iteration %u\n", krylov_idx);
                fflush(dmat_file);
                fflush(bmat_file);
            }
            
# pragma mark Matrix multiplication
            size_t new_max_dets = max_n_dets;
            for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                DistVec<double> &curr_vec = sol_vecs[vec_idx];
                curr_vec.set_curr_vec_idx(0);
                h_op_offdiag(curr_vec, symm, tot_orb, *eris, *h_core, (uint8_t *)orb_indices1, n_frz, n_elec_unf, 1, -eps);
                curr_vec.set_curr_vec_idx(0);
                h_op_diag(curr_vec, 0, 1, -eps);
                curr_vec.add_vecs(0, 1);
                if (curr_vec.max_size() > new_max_dets) {
                    new_max_dets = curr_vec.max_size();
                }
            }
            
            if (new_max_dets > max_n_dets) {
                keep_exact.resize(new_max_dets, false);
                srt_arr.resize(new_max_dets);
            }
#pragma mark Vector compression
            for (uint16_t vec_idx = 0; vec_idx < n_trial; vec_idx++) {
                unsigned int n_samp = target_nonz;
                double glob_norm;
                for (size_t det_idx = 0; det_idx < sol_vecs[vec_idx].curr_size(); det_idx++) {
                    srt_arr[det_idx] = det_idx;
                }
                loc_norms[proc_rank] = find_preserve(sol_vecs[vec_idx].values(), srt_arr.data(), keep_exact, sol_vecs[vec_idx].curr_size(), &n_samp, &glob_norm);
                if (proc_rank == 0) {
                    rn_sys = genrand_mt(rngen_ptr) / (1. + UINT32_MAX);
                }
#ifdef USE_MPI
                MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, loc_norms, 1, MPI_DOUBLE, MPI_COMM_WORLD);
#endif
                sys_comp(sol_vecs[vec_idx].values(), sol_vecs[vec_idx].curr_size(), loc_norms, n_samp, keep_exact, rn_sys);
                for (size_t det_idx = 0; det_idx < sol_vecs[vec_idx].curr_size(); det_idx++) {
                    if (keep_exact[det_idx]) {
                        sol_vecs[vec_idx].del_at_pos(det_idx);
                        keep_exact[det_idx] = 0;
                    }
                }
            }
        }
        //        LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'N', n_trial, krylov_mat.data(), n_trial, energies_r, energies_i, NULL, n_trial, NULL, n_trial);
    }
    
    if (proc_rank == 0) {
        fclose(bmat_file);
        fclose(dmat_file);
    }
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}
