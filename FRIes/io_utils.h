//
//  io_utils.h
//  FRIes
//
//  Created by Samuel Greene on 3/30/19.
//  Copyright © 2019 Samuel Greene. All rights reserved.
//

#ifndef io_utils_h
#define io_utils_h

#include <stdio.h>
#include "csvparser.h"

void read_in_doub(double *buf, char *fname);
void read_in_uchar(unsigned char *buf, char *fname);

#endif /* io_utils_h */
