//
//  compress_1D.c
//  FRIes
//
//  Created by Samuel Greene on 9/25/19.
//  Copyright © 2019 Samuel Greene. All rights reserved.
//

#include <stdio.h>
#include <FRIES/Ext_Libs/dcmt/dc.h>

int main(int argc, const char * argv[]) {
    // Rn generator
    mt_struct *rngen_ptr = get_mt_parameter_id_st(32, 521, 0, 0);
    sgenrand_mt(0, rngen_ptr);
    
    printf("random double: %lf\n", genrand_mt(rngen_ptr) / (1. + UINT32_MAX));
    return 0;
}
