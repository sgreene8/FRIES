/*! \file
 *
 * \brief FRI algorithm with systematic matrix compression for a molecular
 * system
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <FRIES/Hamiltonians/near_uniform.h>
#include <FRIES/io_utils.h>
#include <FRIES/Ext_Libs/dcmt/dc.h>
#include <FRIES/compress_utils.h>
#include <FRIES/Ext_Libs/argparse.h>
#include <FRIES/Hamiltonians/heat_bathPP.h>
#include <FRIES/Hamiltonians/molecule.h>
#define max_iter 1000000

static const char *const usage[] = {
    "frisys_mol [options] [[--] args]",
    "frisys_mol [options]",
    NULL,
};

int main(int argc, const char * argv[]) {
    const char *hf_path = NULL;
    const char *result_dir = "./";
    const char *load_dir = NULL;
    const char *ini_dir = NULL;
    unsigned int target_nonz = 0;
    unsigned int matr_samp = 0;
    unsigned int max_n_dets = 0;
    unsigned int init_thresh = 0;
    unsigned int tmp_norm = 0;
    struct argparse_option options[] = {
        OPT_HELP(),
        OPT_STRING('d', "hf_path", &hf_path, "Path to the directory that contains the HF output files eris.txt, hcore.txt, symm.txt, hf_en.txt, and sys_params.txt"),
        OPT_INTEGER('t', "target", &tmp_norm, "Target one-norm of solution vector"),
        OPT_INTEGER('m', "vec_nonz", &target_nonz, "Target number of nonzero vector elements to keep after each iteration"),
        OPT_INTEGER('M', "mat_nonz", &matr_samp, "Target number of nonzero matrix elements to keep after each iteration"),
        OPT_STRING('y', "result_dir", &result_dir, "Directory in which to save output files"),
        OPT_INTEGER('p', "max_dets", &max_n_dets, "Maximum number of determinants on a single MPI process."),
        OPT_INTEGER('i', "initiator", &init_thresh, "Magnitude of vector element required to make the corresponding determinant an initiator."),
        OPT_STRING('l', "load_dir", &load_dir, "Directory from which to load checkpoint files from a previous systematic FRI calculation (in binary format, see documentation for save_vec())."),
        OPT_STRING('n', "ini_dir", &ini_dir, "Directory from which to read the initial vector for a new calculation (see documentation for load_vec_txt())."),
        OPT_END(),
    };
    
    struct argparse argparse;
    argparse_init(&argparse, options, usage, 0);
    argparse_describe(&argparse, "\nPerform an FCIQMC calculation.", "");
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
    
    double target_norm = tmp_norm;
    
    int n_procs = 1;
    int proc_rank = 0;
    unsigned int proc_idx, hf_proc;
#ifdef USE_MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
#endif
    
    // Parameters
    double shift_damping = 0.05;
    unsigned int shift_interval = 10;
    unsigned int save_interval = 100;
    double en_shift = 0;
    
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
    long long ini_bit = 1LL << (2 * n_orb);
    
    unsigned char *symm = in_data.symm;
    double (* h_core)[tot_orb] = (double (*)[tot_orb])in_data.hcore;
    double (* eris)[tot_orb][tot_orb][tot_orb] = (double (*)[tot_orb][tot_orb][tot_orb])in_data.eris;
    
    // Rn generator
    mt_struct *rngen_ptr = get_mt_parameter_id_st(32, 521, proc_rank, (unsigned int) time(NULL));
    sgenrand_mt((uint32_t) time(NULL), rngen_ptr);
    
    // Solution vector
    unsigned int spawn_length = matr_samp * 2 / n_procs;
    dist_vec *sol_vec = init_vec(max_n_dets, spawn_length, rngen_ptr, n_orb, n_elec_unf, DOUB, 0);
    size_t det_idx;
    
    unsigned char symm_lookup[n_irreps][n_orb + 1];
    gen_symm_lookup(symm, n_orb, n_irreps, symm_lookup);
    unsigned int max_n_symm = 0;
    for (det_idx = 0; det_idx < n_irreps; det_idx++) {
        if (symm_lookup[det_idx][0] > max_n_symm) {
            max_n_symm = symm_lookup[det_idx][0];
        }
    }
    
    // Initialize hash function for processors and vector
    unsigned int proc_scrambler[2 * n_orb];
    double loc_norm, glob_norm;
    double last_norm = 0;
    
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
    sol_vec->proc_scrambler = proc_scrambler;
    
    long long hf_det = gen_hf_bitstring(n_orb, n_elec - n_frz);
    hf_proc = idx_to_proc(sol_vec, hf_det);
    
    unsigned char tmp_orbs[n_elec_unf];
    
    // Calculate HF column vector for energy estimator, and count # single/double excitations
    gen_orb_list(hf_det, sol_vec->tabl, tmp_orbs);
    size_t max_num_doub = count_doub_nosymm(n_elec_unf, n_orb);
    long long *hf_dets = malloc(max_num_doub * sizeof(long long));
    double *hf_mel = malloc(max_num_doub * sizeof(double));
    size_t n_hf_doub = gen_hf_ex(hf_det, tmp_orbs, n_elec_unf, tot_orb, symm, eris, n_frz, hf_dets, hf_mel);
    size_t n_hf_sing = count_singex(hf_det, tmp_orbs, symm, n_orb, symm_lookup, n_elec_unf);
    double p_doub = (double) n_hf_doub / (n_hf_sing + n_hf_doub);
    
    unsigned long long hf_hashes[n_hf_doub];
    for (det_idx = 0; det_idx < n_hf_doub; det_idx++) {
        hf_hashes[det_idx] = idx_to_hash(sol_vec, hf_dets[det_idx]);
    }
    
    // Initialize solution vector
    if (load_dir) {
        load_vec(sol_vec, load_dir);
    }
    else if (ini_dir) {
        long long *load_dets = sol_vec->indices;
        double *load_vals = sol_vec->values;
        
        char buf[100];
        sprintf(buf, "%sfri_", ini_dir);
        size_t n_dets = load_vec_txt(buf, load_dets, load_vals, DOUB);
        
        for (det_idx = 0; det_idx < n_dets; det_idx++) {
            add_doub(sol_vec, load_dets[det_idx], load_vals[det_idx], ini_bit);
        }
    }
    else {
        if (hf_proc == proc_rank) {
            add_doub(sol_vec, hf_det, 100, ini_bit);
        }
    }
    perform_add(sol_vec, ini_bit);
    loc_norm = local_norm(sol_vec);
    sum_mpi_d(loc_norm, &glob_norm, proc_rank, n_procs);
    if (load_dir) {
        last_norm = glob_norm;
    }
    
    char file_path[100];
    FILE *num_file = NULL;
    FILE *den_file = NULL;
    FILE *shift_file = NULL;
    FILE *norm_file = NULL;
    FILE *nkept_file = NULL;
    if (proc_rank == hf_proc) {
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
        strcat(file_path, "nkept.txt");
        nkept_file = fopen(file_path, "a");
        
        strcpy(file_path, result_dir);
        strcat(file_path, "params.txt");
        FILE *param_f = fopen(file_path, "w");
        fprintf(param_f, "FRI calculation\nHF path: %s\nepsilon (imaginary time step): %lf\nTarget norm %lf\nInitiator threshold: %u\nMatrix nonzero: %u\nVector nonzero: %u\n", hf_path, eps, target_norm, init_thresh, matr_samp, target_nonz);
        if (load_dir) {
            fprintf(param_f, "Restarting calculation from %s\n", load_dir);
        }
        else if (ini_dir) {
            fprintf(param_f, "Initializing calculation from vector at path %s\n", ini_dir);
        }
        else {
            fprintf(param_f, "Initializing calculation from HF unit vector\n");
        }
        fclose(param_f);
    }
    
    double *subwt_mem = malloc(sizeof(double) * n_orb * spawn_length);
    unsigned int *ndiv_vec = malloc(sizeof(unsigned int) * spawn_length);
    double *comp_vec1 = malloc(sizeof(double) * spawn_length);
    double *comp_vec2 = malloc(sizeof(double) * spawn_length);
    size_t (*comp_idx)[2] = malloc(sizeof(size_t) * 2 * spawn_length);
    size_t comp_len;
    size_t *det_indices1 = malloc(sizeof(size_t) * spawn_length);
    size_t *det_indices2 = malloc(sizeof(size_t) * spawn_length);
    unsigned char (*orb_indices1)[4] = malloc(sizeof(char) * 4 * spawn_length);
    unsigned char (*orb_indices2)[4] = malloc(sizeof(char) * 4 * spawn_length);
    unsigned int unocc_symm_cts[n_irreps][2];
    int *keep_idx = calloc(n_orb * spawn_length, sizeof(int));
    double *wt_remain = calloc(spawn_length, sizeof(double));
    size_t samp_idx, weight_idx;
    
    hb_info *hb_probs = set_up(tot_orb, n_orb, eris);
    
    double last_one_norm = 0;
    double matr_el;
    double recv_nums[n_procs];
    
    // Parameters for systematic sampling
    double rn_sys = 0;
    double weight;
    int glob_n_nonz; // Number of nonzero elements in whole vector (across all processors)
    double loc_norms[n_procs];
    size_t *srt_arr = malloc(sizeof(size_t) * max_n_dets);
    for (det_idx = 0; det_idx < max_n_dets; det_idx++) {
        srt_arr[det_idx] = det_idx;
    }
    int *keep_exact = calloc(max_n_dets, sizeof(int));
    size_t n_subwt;
    
    unsigned int iterat;
    for (iterat = 0; iterat < max_iter; iterat++) {
        sum_mpi_i(sol_vec->n_nonz, &glob_n_nonz, proc_rank, n_procs);
        
        // Systematic matrix compression
        if (glob_n_nonz > matr_samp) {
            fprintf(stderr, "Warning: target number of matrix samples (%u) is less than number of nonzero vector elements (%d)\n", matr_samp, glob_n_nonz);
        }
        
        // Singles vs doubles
        n_subwt = 2;
        for (det_idx = 0; det_idx < sol_vec->curr_size; det_idx++) {
            double *curr_el = doub_at_pos(sol_vec, det_idx);
            weight = fabs(*curr_el);
            comp_vec1[det_idx] = weight;
            if (weight > 0) {
                subwt_mem[det_idx * n_subwt] = p_doub;
                subwt_mem[det_idx * n_subwt + 1] = (1 - p_doub);
                ndiv_vec[det_idx] = 0;
            }
            else {
                ndiv_vec[det_idx] = 1;
            }
        }
        if (proc_rank == 0) {
            rn_sys = genrand_mt(rngen_ptr) / (1. + UINT32_MAX);
        }
        comp_len = comp_sub(comp_vec1, sol_vec->curr_size, ndiv_vec, n_subwt, (double (*)[n_subwt])subwt_mem, (int (*)[n_subwt])keep_idx, matr_samp, wt_remain, rn_sys, comp_vec2, comp_idx);
        
        // First occupied orbital
        n_subwt = n_elec_unf;
        for (samp_idx = 0; samp_idx < comp_len; samp_idx++) {
            det_idx = comp_idx[samp_idx][0];
            det_indices1[samp_idx] = det_idx;
            orb_indices1[samp_idx][0] = comp_idx[samp_idx][1];
            if (orb_indices1[samp_idx][0] == 0) { // double excitation
                ndiv_vec[samp_idx] = 0;
//                comp_vec2[samp_idx] *= calc_o1_probs(hb_probs, &subwt_mem[samp_idx * n_subwt], n_elec_unf, orbs_at_pos(sol_vec, det_idx));
                calc_o1_probs(hb_probs, &subwt_mem[samp_idx * n_subwt], n_elec_unf, orbs_at_pos(sol_vec, det_idx));
            }
            else {
                unsigned char *occ_orbs = orbs_at_pos(sol_vec, det_idx);
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
            rn_sys = genrand_mt(rngen_ptr) / (1. + UINT32_MAX);
        }
        comp_len = comp_sub(comp_vec2, comp_len, ndiv_vec, n_subwt, (double (*)[n_subwt])subwt_mem, (int (*)[n_subwt])keep_idx, matr_samp, wt_remain, rn_sys, comp_vec1, comp_idx);
        
        // Unoccupied orbital (single); 2nd occupied (double)
        n_subwt = n_elec_unf;
        for (samp_idx = 0; samp_idx < comp_len; samp_idx++) {
            weight_idx = comp_idx[samp_idx][0];
            det_idx = det_indices1[weight_idx];
            det_indices2[samp_idx] = det_idx;
            orb_indices2[samp_idx][0] = orb_indices1[weight_idx][0]; // single or double
            orb_indices2[samp_idx][1] = comp_idx[samp_idx][1]; // first occupied orbital index (converted to orbital below)
            unsigned char *occ_orbs = orbs_at_pos(sol_vec, det_idx);
            if (orb_indices2[samp_idx][0] == 0) { // double excitation
                ndiv_vec[samp_idx] = 0;
//                comp_vec1[samp_idx] *= calc_o2_probs(hb_probs, &subwt_mem[samp_idx * n_subwt], n_elec_unf, occ_orbs, &orb_indices2[samp_idx][1]);
                calc_o2_probs(hb_probs, &subwt_mem[samp_idx * n_subwt], n_elec_unf, occ_orbs, &orb_indices2[samp_idx][1]);
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
            rn_sys = genrand_mt(rngen_ptr) / (1. + UINT32_MAX);
        }
        comp_len = comp_sub(comp_vec1, comp_len, ndiv_vec, n_subwt, (double (*)[n_subwt])subwt_mem, (int (*)[n_subwt])keep_idx, matr_samp, wt_remain, rn_sys, comp_vec2, comp_idx);
        
        // 1st unoccupied (double)
        n_subwt = n_orb;
        for (samp_idx = 0; samp_idx < comp_len; samp_idx++) {
            weight_idx = comp_idx[samp_idx][0];
            det_idx = det_indices2[weight_idx];
            det_indices1[samp_idx] = det_idx;
            orb_indices1[samp_idx][0] = orb_indices2[weight_idx][0]; // single or double
            unsigned char o1_orb = orb_indices2[weight_idx][1];
            orb_indices1[samp_idx][1] = o1_orb; // 1st occupied orbital
            orb_indices1[samp_idx][2] = comp_idx[samp_idx][1]; // 2nd occupied orbital index (doubles), converted to orbital below; unoccupied orbital index (singles)
            if (orb_indices1[samp_idx][0] == 0) { // double excitation
                ndiv_vec[samp_idx] = 0;
                unsigned char *occ_tmp = orbs_at_pos(sol_vec, det_idx);
                orb_indices1[samp_idx][2] = occ_tmp[orb_indices1[samp_idx][2]];
//                comp_vec2[samp_idx] *= calc_u1_probs(hb_probs, &subwt_mem[samp_idx * n_subwt], o1_orb, sol_vec->indices[det_indices1[samp_idx]]);
                calc_u1_probs(hb_probs, &subwt_mem[samp_idx * n_subwt], o1_orb, sol_vec->indices[det_indices1[samp_idx]]);
            }
            else { // single excitation
                orb_indices1[samp_idx][3] = orb_indices2[weight_idx][3];
                ndiv_vec[samp_idx] = 1;
            }
        }
        if (proc_rank == 0) {
            rn_sys = genrand_mt(rngen_ptr) / (1. + UINT32_MAX);
        }
        comp_len = comp_sub(comp_vec2, comp_len, ndiv_vec, n_subwt, (double (*)[n_subwt])subwt_mem, (int (*)[n_subwt])keep_idx, matr_samp, wt_remain, rn_sys, comp_vec1, comp_idx);
        
        // 2nd unoccupied (double)
        n_subwt = max_n_symm;
        for (samp_idx = 0; samp_idx < comp_len; samp_idx++) {
            weight_idx = comp_idx[samp_idx][0];
            det_idx = det_indices1[weight_idx];
            det_indices2[samp_idx] = det_idx;
            orb_indices2[samp_idx][0] = orb_indices1[weight_idx][0]; // single or double
            unsigned char o1_orb = orb_indices1[weight_idx][1];
            orb_indices2[samp_idx][1] = o1_orb; // 1st occupied orbital
            unsigned char o2_orb = orb_indices1[weight_idx][2];
            orb_indices2[samp_idx][2] = o2_orb; // 2nd occupied orbital (doubles); unoccupied orbital index (singles)
            if (orb_indices2[samp_idx][0] == 0) { // double excitation
                ndiv_vec[samp_idx] = 0;
                unsigned char u1_orb = comp_idx[samp_idx][1] + n_orb * (o1_orb / n_orb);
                orb_indices2[samp_idx][3] = u1_orb;
//                comp_vec1[samp_idx] *= calc_u2_probs(hb_probs, &subwt_mem[samp_idx * n_subwt], o1_orb, o2_orb, u1_orb, (unsigned char *)symm_lookup, symm, max_n_symm); // not normalizing
                double u2_norm = calc_u2_probs(hb_probs, &subwt_mem[samp_idx * n_subwt], o1_orb, o2_orb, u1_orb, (unsigned char *)symm_lookup, symm, &max_n_symm);
                if (u2_norm == 0) {
                    comp_vec1[samp_idx] = 0;
                }
            }
            else {
                orb_indices2[samp_idx][3] = orb_indices1[weight_idx][3];
                ndiv_vec[samp_idx] = 1;
            }
        }
        if (proc_rank == 0) {
            rn_sys = genrand_mt(rngen_ptr) / (1. + UINT32_MAX);
        }
        comp_len = comp_sub(comp_vec1, comp_len, ndiv_vec, n_subwt, (double (*)[n_subwt])subwt_mem, (int (*)[n_subwt])keep_idx, matr_samp, wt_remain, rn_sys, comp_vec2, comp_idx);
        
        for (samp_idx = 0; samp_idx < comp_len; samp_idx++) {
            weight_idx = comp_idx[samp_idx][0];
            det_idx = det_indices2[weight_idx];
            long long curr_det = sol_vec->indices[det_idx];
            double *curr_el = doub_at_pos(sol_vec, det_idx);
            long long ini_flag = fabs(*curr_el) > init_thresh;
            ini_flag <<= 2 * n_orb;
            int el_sign = 1 - 2 * (*curr_el < 0);
            unsigned char *occ_orbs = orbs_at_pos(sol_vec, det_idx);
            if (orb_indices2[weight_idx][0] == 0) { // double excitation
                unsigned char doub_orbs[4];
                doub_orbs[0] = orb_indices2[weight_idx][1];
                doub_orbs[1] = orb_indices2[weight_idx][2];
                doub_orbs[2] = orb_indices2[weight_idx][3];
                unsigned char u2_symm = symm[doub_orbs[0] % n_orb] ^ symm[doub_orbs[1] % n_orb] ^ symm[doub_orbs[2] % n_orb];
                doub_orbs[3] = symm_lookup[u2_symm][comp_idx[samp_idx][1] + 1] + n_orb * (doub_orbs[1] / n_orb);
                if (curr_det & (1LL << doub_orbs[3])) { // chosen orbital is occupied; unsuccessful spawn
                    continue;
                }
                if (doub_orbs[2] > doub_orbs[3]) {
                    unsigned char tmp = doub_orbs[3];
                    doub_orbs[3] = doub_orbs[2];
                    doub_orbs[2] = tmp;
                }
                if (doub_orbs[0] > doub_orbs[1]) {
                    unsigned char tmp = doub_orbs[1];
                    doub_orbs[1] = doub_orbs[0];
                    doub_orbs[0] = tmp;
                }
                matr_el = doub_matr_el_nosgn(doub_orbs, tot_orb, eris, n_frz);
                if (fabs(matr_el) > 1e-9 && comp_vec2[samp_idx] > 1e-9) {
//                    matr_el *= -eps / p_doub / calc_unnorm_wt(hb_probs, doub_orbs) * el_sign * par_sign * comp_vec2[samp_idx];
                    matr_el *= -eps / p_doub / calc_norm_wt(hb_probs, doub_orbs, occ_orbs, n_elec_unf, curr_det, (unsigned char *)symm_lookup, symm) * el_sign * comp_vec2[samp_idx];
                    matr_el *= doub_det_parity(&curr_det, doub_orbs);
                    add_doub(sol_vec, curr_det, matr_el, ini_flag);
                }
            }
            else { // single excitation
                unsigned char sing_orbs[2];
                unsigned char o1 = orb_indices2[weight_idx][1];
                sing_orbs[0] = o1;
                unsigned char u1_symm = symm[o1 % n_orb];
                sing_orbs[1] = virt_from_idx(curr_det, symm_lookup[u1_symm], n_orb * (o1 / n_orb), orb_indices2[weight_idx][2]);
                matr_el = sing_matr_el_nosgn(sing_orbs, occ_orbs, tot_orb, eris, h_core, n_frz, n_elec_unf);
                if (fabs(matr_el) > 1e-9 && comp_vec2[samp_idx] > 1e-9) {
                    count_symm_virt(unocc_symm_cts, occ_orbs, n_elec_unf, n_orb, n_irreps, symm_lookup, symm);
                    unsigned int n_occ = count_sing_allowed(occ_orbs, n_elec_unf, symm, n_orb, unocc_symm_cts);
                    matr_el *= -eps / (1 - p_doub) * n_occ * orb_indices2[weight_idx][3] * el_sign * sing_det_parity(&curr_det, sing_orbs) * comp_vec2[samp_idx];
                    add_doub(sol_vec, curr_det, matr_el, ini_flag);
                }
            }
        }
        
        // Death/cloning step
        for (det_idx = 0; det_idx < sol_vec->curr_size; det_idx++) {
            double *curr_el = doub_at_pos(sol_vec, det_idx);
            if (*curr_el != 0) {
                double *diag_el = &(sol_vec->matr_el[det_idx]);
                unsigned char *occ_orbs = orbs_at_pos(sol_vec, det_idx);
                if (isnan(*diag_el)) {
                    *diag_el = diag_matrel(occ_orbs, tot_orb, eris, h_core, n_frz, n_elec) - hf_en;
                }
                *curr_el *= 1 - eps * (*diag_el - en_shift);
            }
        }
        perform_add(sol_vec, ini_bit);
        
        // Compression step
        unsigned int n_samp = target_nonz;
        loc_norms[proc_rank] = find_preserve(sol_vec->values, srt_arr, keep_exact, sol_vec->curr_size, &n_samp, &glob_norm);
        if (proc_rank == hf_proc) {
            fprintf(nkept_file, "%u\n", target_nonz - n_samp);
        }
        
        // Adjust shift
        if ((iterat + 1) % shift_interval == 0) {
            adjust_shift(&en_shift, glob_norm, &last_one_norm, target_norm, shift_damping / shift_interval / eps);
            if (proc_rank == hf_proc) {
                fprintf(shift_file, "%lf\n", en_shift);
                fprintf(norm_file, "%lf\n", glob_norm);
            }
        }
        matr_el = vec_dot(sol_vec, hf_dets, hf_mel, n_hf_doub, hf_hashes);
#ifdef USE_MPI
        MPI_Gather(&matr_el, 1, MPI_DOUBLE, recv_nums, 1, MPI_DOUBLE, hf_proc, MPI_COMM_WORLD);
#else
        recv_nums[0] = matr_el;
#endif
        if (proc_rank == hf_proc) {
            matr_el = 0;
            for (proc_idx = 0; proc_idx < n_procs; proc_idx++) {
                matr_el += recv_nums[proc_idx];
            }
            
            fprintf(num_file, "%lf\n", matr_el);
            double ref_element = ((double *)sol_vec->values)[0];
            fprintf(den_file, "%lf\n", ref_element);
            printf("%6u, en est: %.9lf, shift: %lf, norm: %lf\n", iterat, matr_el / ref_element, en_shift, glob_norm);
        }
        
        if (proc_rank == 0) {
            rn_sys = genrand_mt(rngen_ptr) / (1. + UINT32_MAX);
        }
#ifdef USE_MPI
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, loc_norms, 1, MPI_DOUBLE, MPI_COMM_WORLD);
#endif
        sys_comp(sol_vec->values, sol_vec->curr_size, loc_norms, n_samp, keep_exact, rn_sys);
        for (det_idx = 0; det_idx < sol_vec->curr_size; det_idx++) {
            if (keep_exact[det_idx] && sol_vec->indices[det_idx] != hf_det) {
                del_at_pos(sol_vec, det_idx);
                keep_exact[det_idx] = 0;
            }
        }
        
        if ((iterat + 1) % save_interval == 0) {
            save_vec(sol_vec, result_dir);
            if (proc_rank == hf_proc) {
                fflush(num_file);
                fflush(den_file);
                fflush(shift_file);
                fflush(nkept_file);
            }
        }
    }
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}

