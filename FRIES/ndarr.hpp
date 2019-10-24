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

template <class mat_type>
class Matrix {
public:
    Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols) {
        data_ = new mat_type[rows * cols];
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
    
    mat_type *operator[] (size_t row) const {
        return &data_[cols_ * row];
    }
    
    void enlarge(size_t new_row) {
        data_ = (mat_type *) realloc(data_, sizeof(mat_type) * new_row * cols_);
        rows_ = new_row;
    }
    // ...
    ~Matrix() {                              // Destructor
        delete[] data_;
    }
    Matrix(const Matrix& m);               // Copy constructor
    Matrix& operator= (const Matrix& m);   // Assignment operator
                                           // ...
                                           //private:
    size_t rows_, cols_;
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
      data_ = new double[len1 * len2 * len3 * len4];
    }
    double& operator() (size_t i1, size_t i2, size_t i3, size_t i4);        // Subscript operators often come in pairs
    double  operator() (size_t i1, size_t i2, size_t i3, size_t i4) const {  // Subscript operators often come in pairs
        return data_[i1 * len2_ * len3_ * len4_ + i2 * len3_ * len4_ + i3 * len4_ + i4];
    }
                                                                            // ...
    ~FourDArr() {                              // Destructor
        delete[] data_;
    }
    FourDArr(const FourDArr& m);               // Copy constructor
    FourDArr& operator= (const FourDArr& m);   // Assignment operator
                                               // ...
                                               //private:
    size_t len1_, len2_, len3_, len4_;
    double* data_;
};

#endif /* ndarr_h */
