/*! \file
 *
 * \brief Simple program designed to test linking to FRIES library
 */

#include <iostream>
#include <chrono>
#include <FRIES/compress_utils.hpp

int main(int argc, const char * argv[]) {
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 mt_obj((unsigned int)seed);
    
    int rn = round_binomially(-0.5, 1, mt_obj);
    
    std::cout << "Binomial random number: " << rn << "\n";
    
    return 0;
}
