/*! \file
 *
 * \brief Miscellaneous math utilities and definitions
 */

#include "math_utils.h"


byte_table *gen_byte_table(void) {
    byte_table *new_table = malloc(sizeof(byte_table));
    new_table->nums = malloc(sizeof(uint8_t) * 256);
    new_table->pos = malloc(sizeof(uint8_t) * 256 * 8);
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


unsigned int bits_between(uint8_t *bit_str, uint8_t a, uint8_t b) {
    unsigned int n_bits = 0;
    uint8_t byte_counts[] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};
    uint8_t min_bit, max_bit;
    
    if (a < b) {
        min_bit = a;
        max_bit = b;
    }
    else {
        max_bit = a;
        min_bit = b;
    }
    
    uint8_t mask;
    uint8_t min_byte = min_bit / 8;
    uint8_t max_byte = max_bit / 8;
    int same_byte = min_byte == max_byte;
    uint8_t byte_idx;
    uint8_t curr_int;
    
    byte_idx = min_bit / 8;
    min_bit %= 8;
    max_bit %= 8;
    if (same_byte) {
        mask = (1 << max_bit) - (1 << (min_bit + 1));
    }
    else {
        mask = 256 - (1 << (min_bit + 1));
    }
    curr_int = mask & bit_str[byte_idx];
    n_bits = byte_counts[curr_int & 15];
    curr_int >>= 4;
    n_bits += byte_counts[curr_int & 15];
    for (byte_idx++; byte_idx < max_byte; byte_idx++) {
        curr_int = bit_str[byte_idx];
        n_bits += byte_counts[curr_int & 15];
        curr_int >>= 4;
        n_bits += byte_counts[curr_int & 15];
    }
    if (!same_byte) {
        mask = (1 << max_bit) - 1;
        curr_int = mask & bit_str[byte_idx];
        n_bits += byte_counts[curr_int & 15];
        curr_int >>= 4;
        n_bits += byte_counts[curr_int & 15];
    }
    
    return n_bits;
}


uint8_t find_bits(uint8_t *bit_str, uint8_t *bits, uint8_t n_bytes, byte_table *tabl) {
    unsigned int n_bits = 0;
    uint8_t byte_bits, det_byte, bit_idx;
    for (unsigned int byte_idx = 0; byte_idx < n_bytes; byte_idx++) {
        det_byte = bit_str[byte_idx];
        byte_bits = tabl->nums[det_byte];
        for (bit_idx = 0; bit_idx < byte_bits; bit_idx++) {
            bits[n_bits + bit_idx] = (8 * byte_idx + tabl->pos[det_byte][bit_idx]);
        }
        n_bits += byte_bits;
    }
    return n_bits;
}
