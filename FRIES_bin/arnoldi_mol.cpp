/*! \file
 *
 * \brief FRI applied to Arnoldi method for calculating excited-state energies
 */

#include <cstdio>
#include <ctime>
#include <FRIES/io_utils.hpp>
#include <FRIES/Ext_Libs/dcmt/dc.h>
#include <FRIES/compress_utils.hpp>
#include <FRIES/Ext_Libs/argparse.h>
#include <FRIES/Hamiltonians/heat_bathPP.hpp>
#include <FRIES/Hamiltonians/molecule.hpp>

static const char *const usage[] = {
    "arnoldi_mol [options] [[--] args]",
    "arnoldi_mol [options]",
    NULL,
};


int main(int argc, const char * argv[]) {
    const char *hf_path = NULL;
    const char *result_dir = "./";
    const char *load_dir = NULL;
    const char *ini_path = NULL;
    const char *trial_path = NULL;
    unsigned int n_trial = 0;
    unsigned int target_nonz = 0;
    unsigned int matr_samp = 0;
    unsigned int max_n_dets = 0;
    float init_thresh = 0;
    unsigned int tmp_norm = 0;
    unsigned int max_iter = 1000000;
    struct argparse_option options[] = {
        OPT_HELP(),
        OPT_STRING('d', "hf_path", &hf_path, "Path to the directory that contains the HF output files eris.txt, hcore.txt, symm.txt, hf_en.txt, and sys_params.txt"),
        OPT_INTEGER('t', "target", &tmp_norm, "Target one-norm of solution vector"),
        OPT_INTEGER('m', "vec_nonz", &target_nonz, "Target number of nonzero vector elements to keep after each iteration"),
        OPT_INTEGER('M', "mat_nonz", &matr_samp, "Target number of nonzero matrix elements to keep after each iteration"),
        OPT_STRING('y', "result_dir", &result_dir, "Directory in which to save output files"),
        OPT_INTEGER('p', "max_dets", &max_n_dets, "Maximum number of determinants on a single MPI process."),
        OPT_FLOAT('i', "initiator", &init_thresh, "Magnitude of vector element required to make the corresponding determinant an initiator."),
        OPT_STRING('l', "load_dir", &load_dir, "Directory from which to load checkpoint files from a previous systematic FRI calculation (in binary format, see documentation for DistVec::save() and DistVec::load())."),
        OPT_STRING('n', "ini_vec", &ini_path, "Prefix for files containing the vector with which to initialize the calculation (files must have names <ini_vec>dets and <ini_vec>vals and be text files)."),
        OPT_STRING('v', "trial_vecs", &trial_path, "Prefix for files containing the vectors with which to calculate the energy (files must have names <trial_vec>dets<xx> and <trial_vec>vals<xx>, where xx is a 2-digit number ranging from 0 to (n_trial - 1), and be text files)."),
        OPT_INTEGER('k', "num_trial", &n_trial, "Number of trial vectors to use to calculate dot products with the iterates."),
        OPT_INTEGER('I', "max_iter", &max_iter, "Maximum number of iterations to run the calculation."),
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
    if (trial_path && n_trial == 0) {
        fprintf(stderr, "Error: a path for trial vectors was specified, but the number of vectors to look for at this path was not\n");
        return 0;
    }
    if (!trial_path && n_trial > 0) {
        fprintf(stderr, "Warning: a number of trial vectors was specified, but a path at which to look for them was not. I will just use Hartree-Fock as the trial vector\n");
        n_trial = 0;
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
    
    uint8_t *symm = in_data.symm;
    Matrix<double> *h_core = in_data.hcore;
    FourDArr *eris = in_data.eris;
    
    // Rn generator
    mt_struct *rngen_ptr = get_mt_parameter_id_st(32, 521, proc_rank, (unsigned int) time(NULL));
    sgenrand_mt((uint32_t) time(NULL), rngen_ptr);
    
    // Solution vector
    unsigned int spawn_length = matr_samp * 4 / n_procs;
    size_t adder_size = spawn_length > 1000000 ? 1000000 : spawn_length;
    DistVec<double> sol_vec(max_n_dets, adder_size, rngen_ptr, n_orb * 2, n_elec_unf, n_procs, NULL, NULL, n_trial);
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
        MPI_Bcast(proc_scrambler, 2 * n_orb, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
#endif
    }
    sol_vec.proc_scrambler_ = proc_scrambler;
    
    uint8_t hf_det[det_size];
    gen_hf_bitstring(n_orb, n_elec - n_frz, hf_det);
    hf_proc = sol_vec.idx_to_proc(hf_det);
    
    uint8_t tmp_orbs[n_elec_unf];
    uint8_t (*orb_indices1)[4] = (uint8_t (*)[4])malloc(sizeof(char) * 4 * spawn_length);
    
# pragma mark Set up trial vectors
    std::vector<DistVec<double> *> trial_vecs(n_trial > 0 ? n_trial : 1);
    std::vector<DistVec<double> *> htrial_vecs(n_trial > 0 ? n_trial : 1);
    size_t n_ex = n_orb * n_orb * n_elec_unf * n_elec_unf;
    if (n_trial > 0) {
        unsigned int n_trial_dets = 0;
        Matrix<uint8_t> &load_dets = sol_vec.indices();
        double *load_vals = (double *)sol_vec.values();
        
        char vec_path[300];
        for (unsigned int trial_idx = 0; trial_idx < n_trial; trial_idx++) {
            if (proc_rank == 0) {
                sprintf(vec_path, "%s%02d", trial_path, trial_idx);
                n_trial_dets = (unsigned int) load_vec_txt(vec_path, load_dets, load_vals, DOUB);
            }
#ifdef USE_MPI
            MPI_Bcast(&n_trial_dets, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
#endif
            trial_vecs[trial_idx] = new DistVec<double>(n_trial_dets, n_trial_dets, rngen_ptr, n_orb * 2, n_elec_unf, n_procs);
            htrial_vecs[trial_idx] = new DistVec<double>(n_trial_dets * n_ex / n_procs, n_trial_dets * n_ex / n_procs, rngen_ptr, n_orb * 2, n_elec_unf, n_procs);
            trial_vecs[trial_idx]->proc_scrambler_ = proc_scrambler;
            htrial_vecs[trial_idx]->proc_scrambler_ = proc_scrambler;
            
            for (det_idx = 0; det_idx < n_trial_dets; det_idx++) {
                trial_vecs[trial_idx]->add(load_dets[det_idx], load_vals[det_idx], 1);
                htrial_vecs[trial_idx]->add(load_dets[det_idx], load_vals[det_idx], 1);
            }
        }
    }
    else {
        trial_vecs[0] = new DistVec<double>(1, 1, rngen_ptr, n_orb * 2, n_elec_unf, n_procs);
        htrial_vecs[0] = new DistVec<double>(n_ex / n_procs, n_ex / n_procs, rngen_ptr, n_orb * 2, n_elec_unf, n_procs);
        trial_vecs[0]->proc_scrambler_ = proc_scrambler;
        htrial_vecs[0]->proc_scrambler_ = proc_scrambler;
        trial_vecs[0]->add(hf_det, 1, 1);
        htrial_vecs[0]->add(hf_det, 1, 1);
        n_trial = 1;
    }
    uintmax_t **trial_hashes = (uintmax_t **)malloc(sizeof(uintmax_t *) * n_trial);
    uintmax_t **htrial_hashes = (uintmax_t **)malloc(sizeof(uintmax_t *) * n_trial);
    for (unsigned int trial_idx = 0; trial_idx < n_trial; trial_idx++) {
        DistVec<double> *curr_trial = trial_vecs[trial_idx];
        curr_trial->perform_add();
        curr_trial->collect_procs();
        trial_hashes[trial_idx] = (uintmax_t *)malloc(sizeof(uintmax_t) * curr_trial->curr_size());
        for (det_idx = 0; det_idx < curr_trial->curr_size(); det_idx++) {
            trial_hashes[trial_idx][det_idx] = sol_vec.idx_to_hash(curr_trial->indices()[det_idx], tmp_orbs);
        }
        
        DistVec<double> * curr_htrial = htrial_vecs[trial_idx];
        curr_htrial->perform_add();
        h_op(*curr_htrial, symm, tot_orb, *eris, *h_core, (uint8_t *)orb_indices1, n_frz, n_elec_unf, 0, 1, hf_en);
        curr_htrial->collect_procs();
        htrial_hashes[trial_idx] = (uintmax_t *)malloc(sizeof(uintmax_t) * curr_htrial->curr_size());
        for (det_idx = 0; det_idx < curr_htrial->curr_size(); det_idx++) {
            htrial_hashes[trial_idx][det_idx] = sol_vec.idx_to_hash(curr_htrial->indices()[det_idx], tmp_orbs);
        }
    }
    
    // Count # single/double excitations from HF
    sol_vec.gen_orb_list(hf_det, tmp_orbs);
    size_t n_hf_doub = doub_ex_symm(hf_det, tmp_orbs, n_elec_unf, n_orb, orb_indices1, symm);
    size_t n_hf_sing = count_singex(hf_det, tmp_orbs, symm, n_orb, symm_lookup, n_elec_unf);
    double p_doub = (double) n_hf_doub / (n_hf_sing + n_hf_doub);
    
    char file_path[300];
    FILE *en_file = NULL;
    FILE *shift_file = NULL;
    
    
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}
