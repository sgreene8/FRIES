//
//  io_utils.c
//  FRIes
//
//  Created by Samuel Greene on 3/30/19.
//  Copyright © 2019 Samuel Greene. All rights reserved.
//

#include "io_utils.h"

void read_in_doub(double *buf, char *fname) {
    int i =  0;
    size_t row_idx = 0;
    size_t n_col;
    //                                   file, delimiter, first_line_is_header?
    CsvParser *csvparser = CsvParser_new(fname, ",", 0);
    CsvRow *row;
    
    while ((row = CsvParser_getRow(csvparser)) ) {
        const char **rowFields = CsvParser_getFields(row);
        n_col = CsvParser_getNumFields(row);
        for (i = 0 ; i < n_col ; i++) {
            sscanf(rowFields[i], "%lf", &buf[n_col * row_idx + i]);
        }
        CsvParser_destroy_row(row);
        row_idx++;
    }
    CsvParser_destroy(csvparser);
}

void read_in_uchar(unsigned char *buf, char *fname) {
    int i =  0;
    size_t row_idx = 0;
    size_t n_col;
    //                                   file, delimiter, first_line_is_header?
    CsvParser *csvparser = CsvParser_new(fname, " ", 0);
    CsvRow *row;
    unsigned int scan_val;
    
    while ((row = CsvParser_getRow(csvparser)) ) {
        const char **rowFields = CsvParser_getFields(row);
        n_col = CsvParser_getNumFields(row);
        for (i = 0 ; i < n_col ; i++) {
            sscanf(rowFields[i], "%u", &scan_val);
            buf[n_col * row_idx + i] = scan_val;
        }
        CsvParser_destroy_row(row);
        row_idx++;
    }
    CsvParser_destroy(csvparser);
}
