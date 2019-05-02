//
//  compress_utils.h
//  FRIes
//
//  Created by Samuel Greene on 4/13/19.
//  Copyright © 2019 Samuel Greene. All rights reserved.
//

#ifndef compress_utils_h
#define compress_utils_h

#include <stdio.h>
#include "dc.h"
#include <math.h>


/* Round p to integer b such that
 b ~ binomial(n, p - floor(p)) + floor(p) * n
 
 Parameters
 ----------
 p: non-integer number to be rounded
 n: Number of rn's to sample
 mt_ptr: Address to MT state object to use for RN generation
 
 Returns
 -------
 integer result
 
 */
int round_binomially(double p, unsigned int n, mt_struct *mt_ptr);

#endif /* compress_utils_h */
