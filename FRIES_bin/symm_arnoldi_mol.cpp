/*! \file
 *
 * \brief FRI applied to Arnoldi method for calculating excited-state energies
 * Rayleigh quotients are constructed from the same random vector on the L & R sides
 */

#include <cstdio>
#include <iostream>
#include <ctime>
#include <FRIES/io_utils.hpp>
#include <FRIES/Ext_Libs/dcmt/dc.h>
#include <FRIES/compress_utils.hpp>
#include <FRIES/Ext_Libs/argparse.hpp>
#include <FRIES/Hamiltonians/molecule.hpp>
#include <stdexcept>

struct MyArgs : public argparse::Args {
    std::string hf_path = kwarg("hf_path", "Path to the directory that contains the HF output files eris.txt, hcore.txt, symm.txt, hf_en.txt, and sys_params.txt");
    uint32_t max_iter = kwarg("max_iter", "Maximum number of iterations to run the calculation").set_default(1000000);
    uint32_t target_nonz = kwarg("vec_nonz", "Target number of nonzero vector elements to keep after each iteration");
    std::string result_dir = kwarg("result_dir", "Directory in which to save output files").set_default<std::string>("./");
    uint32_t max_n_dets = kwarg("max_dets", "Maximum number of determinants on a single MPI process");
    std::string ini_path = kwarg("ini_vecs", "Prefix for files containing the vectors with which to initialize the calculation. Files must have names <ini_vecs>dets<xx> and <ini_vecs>vals<xx>, where xx is a 2-digit number ranging from 0 to (num_states - 1), and be text files");
    uint8_t n_states = kwarg("num_states", "Number of states whose energies will be computed");
    uint16_t n_krylov = kwarg("n_krylov", "Number of multiplications by (1 - \eps H) to include in each iteration").set_default(1000);

    CONSTRUCTOR(MyArgs);
};

int main(int argc, char * argv[]) {
    MyArgs args(argc, argv);
    
    uint8_t n_states = args.n_states;
    if (n_states < 2) {
        fprintf(stderr, "Warning: n_states is 1 or 0. Consider using the power method instead of Arnoldi in this case.\n");
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
    parse_hf_input(args.hf_path.c_str(), &in_data);
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
            proc_scrambler[det_idx] = genrand_mt(rngen_ptr);
        }
        save_proc_hash(args.result_dir.c_str(), proc_scrambler.data(), 2 * n_orb);
    }
#ifdef USE_MPI
    MPI_Bcast(proc_scrambler.data(), 2 * n_orb, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
#endif
    
    std::vector<uint32_t> vec_scrambler(2 * n_orb);
    for (size_t det_idx = 0; det_idx < 2 * n_orb; det_idx++) {
        vec_scrambler[det_idx] = genrand_mt(rngen_ptr);
    }
    
    std::vector<DistVec<double>> sol_vecs;
    sol_vecs.reserve(n_states);
    // vector 0: current iteration before multiplication
    // vector 1: current iteration after multiplication
    // vector 2: temporary vector for off-diagonal multiplication
    // vector 3: initial vector
    for (uint8_t vec_idx = 0; vec_idx < n_states; vec_idx++) {
        sol_vecs.emplace_back(max_n_dets, adder_size, n_orb * 2, n_elec_unf, n_procs, diag_shortcut, nullptr, 4, proc_scrambler, vec_scrambler);
    }
    size_t det_size = CEILING(2 * n_orb, 8);
    
    uint8_t tmp_orbs[n_elec_unf];
    uint8_t (*orb_indices1)[4] = (uint8_t (*)[4])malloc(sizeof(char) * 4 * spawn_length);
    
# pragma mark Set up vectors
    char vec_path[300];
    Matrix<uint8_t> *load_dets = new Matrix<uint8_t>(args.max_n_dets, det_size);
    for (unsigned int trial_idx = 0; trial_idx < n_states; trial_idx++) {
        DistVec<double> &curr_sol = sol_vecs[trial_idx];
        double *load_vals = curr_sol.values();
        
        sprintf(vec_path, "%s%02d", args.ini_path.c_str(), trial_idx);
        unsigned int loc_n_dets = (unsigned int) load_vec_txt(vec_path, *load_dets, load_vals, DOUB);
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
    
    char file_path[300];
    FILE *bmat_file = NULL;
    FILE *dmat_file = NULL;
    
    if (proc_rank == 0) {
        // Setup output files
        strcpy(file_path, args.result_dir.c_str());
        strcat(file_path, "b_matrix.txt");
        bmat_file = fopen(file_path, "a");
        if (!bmat_file) {
            fprintf(stderr, "Could not open file for writing in directory %s\n", args.result_dir.c_str());
        }
        
        strcpy(file_path, args.result_dir.c_str());
        strcat(file_path, "d_matrix.txt");
        dmat_file = fopen(file_path, "a");
        
        strcpy(file_path, args.result_dir.c_str());
        strcat(file_path, "params.txt");
        FILE *param_f = fopen(file_path, "w");
        fprintf(param_f, "Arnoldi calculation\nHF path: %s\nepsilon (imaginary time step): %lf\nVector nonzero: %u\n", args.hf_path.c_str(), eps, args.target_nonz);
        fprintf(param_f, "Path for initial vectors: %s\n", args.ini_path.c_str());
        fprintf(param_f, "Krylov iterations: %u\n", args.n_krylov);
        fclose(param_f);
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
    
    for (uint32_t iteration = 0; iteration < args.max_iter; iteration++) {
        // Initialize the solution vectors
        for (uint16_t vec_idx = 0; vec_idx < n_states; vec_idx++) {
            sol_vecs[vec_idx].copy_vec(3, 0);
        }
        if (proc_rank == 0) {
            printf("Macro iteration %u\n", iteration);
        }
        
        for (uint16_t krylov_idx = 0; krylov_idx < args.n_krylov; krylov_idx++) {
            if (proc_rank == 0) {
                printf("Krylov iteration %u\n", krylov_idx);
                fflush(dmat_file);
                fflush(bmat_file);
            }
            
# pragma mark Matrix multiplication
            for (uint16_t vec_idx = 0; vec_idx < n_states; vec_idx++) {
                DistVec<double> &curr_vec = sol_vecs[vec_idx];
                curr_vec.copy_vec(0, 1);
                curr_vec.set_curr_vec_idx(1);
                h_op_offdiag(curr_vec, symm, tot_orb, *eris, *h_core, (uint8_t *)orb_indices1, n_frz, n_elec_unf, 2, -eps);
                curr_vec.set_curr_vec_idx(1);
                h_op_diag(curr_vec, 1, 1, -eps);
                curr_vec.add_vecs(1, 2);
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
                    // <v|(1 - \eps H)|v> = <v|v> - \eps <v|H|v>
                    vec_H_ovlp = -(vec_H_ovlp - vec_ovlp) / eps;
                    if (proc_rank == 0) {
                        fprintf(dmat_file, "%.9lf,", vec_ovlp);
                        fprintf(bmat_file, "%.9lf,", vec_H_ovlp);
                    }
                }
                if (proc_rank == 0) {
                    fprintf(dmat_file, "\n");
                    fprintf(bmat_file, "\n");
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
        }
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
