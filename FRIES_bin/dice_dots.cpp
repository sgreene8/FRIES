/*! \file
 *
 * \brief A Simple utility for calculating dot products of vectors from Dice output files
 */

#include <iostream>
#include <FRIES/Ext_Libs/argparse.hpp>
#include <FRIES/vec_utils.hpp>

struct MyArgs : public argparse::Args {
    std::string &file_1 = kwarg("file1", "Path to the file containing the first set of vectors");
    std::string &file_2 = kwarg("file2", "Path to the file containing the second set of vectors");
    uint32_t &nvec_1 = kwarg("nvec1", "Number of vectors to read from the first file");
    uint32_t &nvec_2 = kwarg("nvec2", "Number of vectors to read from the second file");
    uint32_t &n_elec = kwarg("n_elec", "Number of electrons in the chemical system under consideration");
    uint32_t &n_orb = kwarg("n_orb", "Number of orbitals in the chemical system under consideration");
};

int main(int argc, char * argv[]) {
    MPI_Init(NULL, NULL);
    MyArgs args = argparse::parse<MyArgs>(argc, argv);
    
    size_t max_el = 100000;

    std::mt19937 mt_obj(0);
    std::vector<uint32_t> proc_scrambler(2 * args.n_orb);
    std::vector<uint32_t> vec_scrambler(2 * args.n_orb);
    for (size_t proc_idx = 0; proc_idx < 2 * args.n_orb; proc_idx++) {
        proc_scrambler[proc_idx] = mt_obj();
        vec_scrambler[proc_idx] = mt_obj();
    }
    
    
    DistVec<double> trial_vecs(max_el, max_el, args.n_orb * 2, args.n_elec, 1, nullptr, args.nvec_1 + 1, proc_scrambler, vec_scrambler);

    size_t det_size = CEILING(2 * args.n_orb, 8);
    Matrix<uint8_t> load_dets(max_el, det_size);
    trial_vecs.set_curr_vec_idx(args.nvec_1);
    double *load_vals = trial_vecs.values();
    for (uint8_t vec_idx = 0; vec_idx < args.nvec_1; vec_idx++) {
        size_t loc_n_dets = load_vec_dice(args.file_1, load_dets, load_vals, vec_idx, args.n_orb);
        for (size_t det_idx = 0; det_idx < loc_n_dets; det_idx++) {
            trial_vecs.add(load_dets[det_idx], load_vals[det_idx], 1);
        }
        trial_vecs.set_curr_vec_idx(vec_idx);
        trial_vecs.perform_add(0);
    }
    
    Matrix<double> d_prods(args.nvec_1, args.nvec_2);
    for (uint8_t vec_idx2 = 0; vec_idx2 < args.nvec_2; vec_idx2++) {
        size_t loc_n_dets = load_vec_dice(args.file_2, load_dets, load_vals, vec_idx2, args.n_orb);
        for (size_t det_idx = 0; det_idx < loc_n_dets; det_idx++) {
            trial_vecs.add(load_dets[det_idx], load_vals[det_idx], 1);
        }
        trial_vecs.set_curr_vec_idx(args.nvec_1);
        trial_vecs.perform_add(0);
        for (uint8_t vec_idx1 = 0; vec_idx1 < args.nvec_1; vec_idx1++) {
            d_prods(vec_idx1, vec_idx2) = trial_vecs.internal_dot(vec_idx1, args.nvec_1);
        }
    }
    
    for (uint8_t row_idx = 0; row_idx < args.nvec_1; row_idx++) {
        for (uint8_t col_idx = 0; col_idx < args.nvec_2 - 1; col_idx++) {
            std::cout << d_prods(row_idx, col_idx) << ",";
        }
        std::cout << d_prods(row_idx, args.nvec_2 - 1) << "\n";
    }

    MPI_Finalize();
    return 0;
}
