/*! \file C++ definitions of classes for variable-length multidimensional arrays
 * Largely copied from https://isocpp.org/wiki/faq/operator-overloading#matrix-subscript-op
 */

#ifndef ndarr_h
#define ndarr_h

#include <stdlib.h>
#include <stdio.h>

/*! \brief A class for representing and manipulating matrices with variable dimension sizes
 * \tparam mat_type The type of elements to be stored in the matrix
 */
template <class mat_type>
class Matrix {
public:
    /*! \brief Constructor for Matrix class
    * \param [in] rows     The number of rows the matrix should have initially
    * \param [in] cols     The number of columns the matrix should have initially
     */
    Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols), tot_size_(rows * cols) {
        data_ = (mat_type *)malloc(sizeof(mat_type) * tot_size_);
    }
    
    /*! \brief Access matrix element
     * \param [in] row      Row index of element
     * \param [in] col      Column index of element
     * \return Reference to matrix element
     */
    mat_type& operator() (size_t row, size_t col) {
        return data_[cols_*row + col];
    }
    
    /*! \brief Access matrix element
     * \param [in] row      Row index of element
     * \param [in] col      Column index of element
     * \return matrix element
     */
    mat_type  operator() (size_t row, size_t col) const {
        return data_[cols_ * row + col];
    }
    
    /*! \brief Access matrix row
     * \param [in] row      Row index
     * \return pointer to 0th element in a row of a matrix
     */
    mat_type *operator[] (size_t row) {
        return &data_[cols_ * row];
    }
    
    /*! \brief Access matrix row
     * \param [in] row      Row index
     * \return pointer to 0th element in a row of a matrix
     */
    const mat_type *operator[] (size_t row) const {
        return &data_[cols_ * row];
    }
    
    /*! \brief Increase number of rows in matrix
     * \param [in] new_row      Desired number of rows in matrix
     */
    void enlarge(size_t new_row) {
        tot_size_ = new_row * cols_;
        data_ = (mat_type *) realloc(data_, sizeof(mat_type) * tot_size_);
        rows_ = new_row;
    }
    
    /*! \brief Change the dimensions without changing any of the data
     * \param [in] new_cols     Desired number of columns in the reshaped matrix
     */
    void reshape(size_t new_cols) {
        rows_ = tot_size_ / new_cols;
        cols_ = new_cols;
    }
    
    /*! \brief Destructor*/
    ~Matrix() {
        free(data_);
    }
    
    Matrix(const Matrix& m) = delete;
    Matrix& operator= (const Matrix& m) = delete;
    
    /*! \return Current number of rows in matrix */
    size_t rows() const {
        return rows_;
    }
    /*! \return Current number of columns in matrix*/
    size_t cols() const {
        return cols_;
    }
private:
    size_t rows_, cols_, tot_size_;
    mat_type* data_;
};


/*! \brief A class for storing 4-D arrays of doubles*/
class FourDArr {
public:
    /*! \brief Constructor
    * \param [in] len1 len2 len3 len4    Lengths of the 4 dimensions of the array
     */
    FourDArr(size_t len1, size_t len2, size_t len3, size_t len4)
    : len1_(len1)
    , len2_(len2)
    , len3_(len3)
    , len4_(len4)
    {
        data_ = (double *)malloc(sizeof(double) * len1 * len2 * len3 * len4);
    }
    
    /*! \brief Access an element of the 4-D array
    * \param [in] i1 First index
    * \param [in] i2 Second index
    * \param [in] i3 Third index
    * \param [in] i4 Fourth index
     * \returns Reference to array element
     */
    double& operator() (size_t i1, size_t i2, size_t i3, size_t i4) {
      return data_[i1 * len2_ * len3_ * len4_ + i2 * len3_ * len4_ + i3 * len4_ + i4];
    }
    
    /*! \brief Access an element of the 4-D array
    * \param [in] i1 First index
    * \param [in] i2 Second index
    * \param [in] i3 Third index
    * \param [in] i4 Fourth index
     * \returns Array element
     */
    double  operator() (size_t i1, size_t i2, size_t i3, size_t i4) const {
        return data_[i1 * len2_ * len3_ * len4_ + i2 * len3_ * len4_ + i3 * len4_ + i4];
    }
    
    /*! \brief Destructor*/
    ~FourDArr() {
        free(data_);
    }
    
    FourDArr(const FourDArr& m) = delete;
    
    FourDArr& operator= (const FourDArr& m) = delete;
    
    /*! \returns pointer to 0th element in the array */
    double *data() {
        return data_;
    }
private:
    size_t len1_, len2_, len3_, len4_; ///< Dimensions of the array
    double* data_; ///< The data stored in the array
};

#endif /* ndarr_h */
