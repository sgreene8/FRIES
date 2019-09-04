/*! \file
 *
 * \brief Miscellaneous math utilities and definitions
 */

#include "math_utils.h"


byte_table *gen_byte_table(void) {
    byte_table *new_table = malloc(sizeof(byte_table));
    new_table->nums = malloc(sizeof(unsigned char) * 256);
    new_table->pos = malloc(sizeof(unsigned char) * 256 * 8);
    unsigned int byte;
    unsigned int bit;
    unsigned int num;
    for (byte = 0; byte < 256; byte++) {
        num = 0;
        for (bit = 0; bit < 8; bit++) {
            if (byte & (1 << bit)) {
                new_table->pos[byte][num] = bit;
                num++;
            }
        }
        new_table->nums[byte] = num;
    }
    return new_table;
}


unsigned int count_bits(long long num, byte_table *tab) {
    unsigned int n_bits = 0;
    unsigned char curr_byte;
    while (num != 0) {
        curr_byte = num & 255;
        n_bits += tab->nums[curr_byte];
        num >>= 8;
    }
    return n_bits;
}


unsigned int bits_between(long long bit_str, unsigned char a, unsigned char b) {
    // count number of 1's between bits a and b in binary representation of bit_str
    unsigned int n_bits = 0;
    unsigned char byte_counts[] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};
    unsigned char min_bit, max_bit;
    
    if (a < b) {
        min_bit = a;
        max_bit = b;
    }
    else {
        max_bit = a;
        min_bit = b;
    }
    
    long long mask = (1LL << max_bit) - (1LL << (min_bit + 1));
    long long curr_int = (bit_str & mask) >> (min_bit + 1);
    
    while (curr_int != 0) {
        n_bits += byte_counts[curr_int & 15];
        curr_int >>= 4;
    }
    
    return n_bits;
}
