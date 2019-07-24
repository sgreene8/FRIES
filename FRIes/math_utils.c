//
//  math_utils.c
//  FRIes
//
//  Created by Samuel Greene on 7/18/19.
//  Copyright © 2019 Samuel Greene. All rights reserved.
//

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
