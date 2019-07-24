//
//  math_utils.h
//  FRIes
//
//  Created by Samuel Greene on 7/18/19.
//  Copyright © 2019 Samuel Greene. All rights reserved.
//

#ifndef math_utils_h
#define math_utils_h

#include <stdio.h>
#include <stdlib.h>

typedef struct {
    unsigned char (*pos)[8]; // 256 x 8
    unsigned char *nums; // 256
} byte_table;

typedef enum {
    DOUB,
    INT
} dtype;

/* Generate lookup table used to decompose a byte into a list of positions of 1's (for 0-255).
 
 Returns
 -------
 byte_table structure containing pointer to an array containing the positions of 1's
 in each byte, and an array containing the number of 1's in the binary representation
 of each byte
 */
byte_table *gen_byte_table(void);

/*
 Count the number of 1's in the binary representation of a number
 */
unsigned int count_bits(long long num, byte_table *tab);

#endif /* math_utils_h */
