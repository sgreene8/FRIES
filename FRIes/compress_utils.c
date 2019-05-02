//
//  compress_utils.c
//  FRIes
//
//  Created by Samuel Greene on 4/13/19.
//  Copyright © 2019 Samuel Greene. All rights reserved.
//

#include "compress_utils.h"


int round_binomially(double p, unsigned int n, mt_struct *mt_ptr) {
    int flr = floor(p);
    double prob = p - flr;
    int ret_val = flr * n;
    unsigned int i;
    for (i = 0; i < n; i++) {
        ret_val += (genrand_mt(mt_ptr) / MT_MAX) < prob;
    }
    return ret_val;
}
