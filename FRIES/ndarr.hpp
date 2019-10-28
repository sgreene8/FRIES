//
//  ndarr.hpp
//  FRIes
//
//  Created by Samuel Greene on 10/23/19.
//  Copyright © 2019 Samuel Greene. All rights reserved.
//

#ifndef ndarr_h
#define ndarr_h

#include <stdlib.h>
#include <stdio.h>

template <class mat_type>
class Matrix {
public:
    Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols), tot_size_(rows * cols) {
        data_ = (mat_type *)malloc(sizeof(mat_type) * tot_size_);
    }
    
    mat_type& operator() (size_t row, size_t col) {        // Subscript operators often come in pairs
        return data_[cols_*row + col];
    }
    mat_type  operator() (size_t row, size_t col) const {  // Subscript operators often come in pairs
        return data_[cols_ * row + col];
    }
    
    mat_type *operator[] (size_t row) {
        return &data_[cols_ * row];
    }
    
    const mat_type *operator[] (size_t row) const {
        return &data_[cols_ * row];
    }
    
    mat_type *data() {
        return data_;
    }
    
    void enlarge(size_t new_row) {
        tot_size_ = new_row * cols_;
        data_ = (mat_type *) realloc(data_, sizeof(mat_type) * tot_size_);
        rows_ = new_row;
    }
    
    void reshape(size_t new_cols) {
        rows_ = tot_size_ / new_cols;
        cols_ = new_cols;
    }
    
    ~Matrix() {                              // Destructor
        free(data_);
    }
    Matrix(const Matrix& m) = delete;               // Copy constructor
    Matrix& operator= (const Matrix& m) = delete;   // Assignment operator
    size_t rows() {
        return rows_;
    }
private:
    size_t rows_, cols_, tot_size_;
    mat_type* data_;
};

class FourDArr {
public:
    FourDArr(size_t len1, size_t len2, size_t len3, size_t len4)
    : len1_(len1)
    , len2_(len2)
    , len3_(len3)
    , len4_(len4)
    {
        data_ = (double *)malloc(sizeof(double) * len1 * len2 * len3 * len4);
    }
    double& operator() (size_t i1, size_t i2, size_t i3, size_t i4);        // Subscript operators often come in pairs
    double  operator() (size_t i1, size_t i2, size_t i3, size_t i4) const {  // Subscript operators often come in pairs
        return data_[i1 * len2_ * len3_ * len4_ + i2 * len3_ * len4_ + i3 * len4_ + i4];
    }
    ~FourDArr() {                              // Destructor
        free(data_);
    }
    FourDArr(const FourDArr& m) = delete;               // Copy constructor
    FourDArr& operator= (const FourDArr& m) = delete;   // Assignment operator
    
    double *data() {
        return data_;
    }
private:
    size_t len1_, len2_, len3_, len4_;
    double* data_;
};

#endif /* ndarr_h */
