/*! \file
 * \brief Test for compression or sampling algorithms
 *
 * This program repeatedly performs a given sampling or compression operation and calculates the maximum
 * deviation between the cumulative mean of the compressed vector (samples) and the input vector (target distribution).
 *
 * This deviation should decrease according to the Central Limit Theorem (n^-1/2), where n is the number of
 * compressions or samples drawn
 *
 * A plot of log(maximum deviation) vs log(number of samples) should have a slope of approximately -1/2
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <FRIES/compress_utils.hpp>
#include "sampler.hpp"

int main(int argc, const char * argv[]) {
    int proc_rank = 0;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    
    unsigned int n_iter = 10000;
    std::ofstream out_file;
    if (proc_rank == 0) {
        out_file.open("max_diff.txt");
    }
//    ParBudget test(5);
//    SysSerial test(10);
    PivSerial test(10);
//    SysStratified test(10, 4 * n_procs - 1);
    
    for (unsigned int iter = 0; iter < n_iter; iter++) {
        test.sample();
        double diff = test.calc_max_diff();
        if (proc_rank == 0) {
            out_file << diff << "\n";
        }
    }


    MPI_Finalize();
    return 0;
}
