/*! \file
 * \brief Test for compression or sampling algorithms
 *
 * This program repeatedly performs a given sampling or compression operation and calculates the maximum
 * deviation between the cumulative mean of the compressed vector (samples) and the input vector (target distribution).
 *
 * This deviation should decreases according to the Central Limit Theorem (n^-1/2), where n is the number of
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
    unsigned int n_iter = 2000;
    std::ofstream out_file;
    out_file.open("max_diff.txt");
    NonuniComp test(100, 30);
    
    for (unsigned int iter = 0; iter < n_iter; iter++) {
        test.sample();
        out_file << test.calc_max_diff() << "\n";
    }
}
