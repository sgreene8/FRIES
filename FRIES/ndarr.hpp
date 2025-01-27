/*! \file C++ definitions of classes for variable-length multidimensional arrays
 * Largely copied from https://isocpp.org/wiki/faq/operator-overloading#matrix-subscript-op
 */

#ifndef ndarr_h
#define ndarr_h

#include <vector>
#include <cstdlib>
#include <FRIES/math_utils.h>

/*! \brief A class for representing and manipulating matrices with variable dimension sizes
 * \tparam mat_type The type of elements to be stored in the matrix
 */
template <class mat_type>
class Matrix {
public:
    /*! \brief A nested class for referencing rows within the Matrix */
    class RowReference {
        private:
            /*! \brief Index of the row */
            size_t row_idx_;
            /*! \brief Reference to the parent matrix */
            Matrix<mat_type> &mat_;
            
        public:
        RowReference(Matrix<mat_type> &mat, size_t row) : row_idx_(row), mat_(mat) {}
        
        mat_type& operator[] (size_t idx) {
            return mat_(row_idx_, idx);
        }
    };
    
    /*! \brief Constructor for Matrix class
    * \param [in] rows     The number of rows the matrix should have initially
    * \param [in] cols     The number of columns the matrix should have initially
    */
    Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols), tot_size_(rows * cols), data_(rows * cols) {}
    
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
    
    /*! \brief Zero all matrix elements */
    void zero() {
        std::fill(data_.begin(), data_.end(), 0);
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

    /*! \brief Remove row from matrix
     * Data are copied such that the first n elements in each row remain the same before and after this operation
     * \param [in] new_col      Desired number of columns in the enlarged matrix
     * \param [in] n_keep       Number of elements to preserve in all rows of the matrix
     */
    void remove_row(size_t row, int rank) { 
        //std::cout << "rows_: " << rows_ << " row: " << row << "\n";

        if ((row < rows_) && (row >= 0)) {
            for (size_t row_idx = 0; row_idx < row; row_idx++) {
                for (size_t col_idx = 0; col_idx < cols_; col_idx++) {
                    (*this)(row_idx, col_idx) = (*this)(row_idx, col_idx);
                }
            }
            for (size_t row_idx = (size_t)(row+1); row_idx < rows_; row_idx++) {
                for (size_t col_idx = 0; col_idx < cols_; col_idx++) {
                    (*this)((size_t)(row_idx-1), col_idx) = (*this)(row_idx, col_idx);
                }
            }
            reshape((rows_ - 1), cols_);
        }
        else {
            std::cout << "ERROR: Attempted to remove row out of bounds for the matrix - row " << row << " on rank " << rank << "\n";
            exit(0);
        }
    }

    /*! \brief Add row to matrix
     * Data are copied such that the first n elements in each row remain the same before and after this operation
     * \param [in] new_col      Desired number of columns in the enlarged matrix
     * \param [in] n_keep       Number of elements to preserve in all rows of the matrix
     */
    void add_row(size_t row, int rank) { 
        //std::cout << "rows_: " << rows_ << " row: " << row << "\n";
        reshape((rows_ + 1), cols_);

        if ((row < rows_) && (row >= 0)) {
            for (size_t row_idx = 0; row_idx < row; row_idx++) {
                for (size_t col_idx = 0; col_idx < cols_; col_idx++) {
                    (*this)(row_idx, col_idx) = (*this)(row_idx, col_idx);
                }
            }

            //std::cout << "rank: " << rank << " vacancies_pos.print(): " << "\n"; 
            //(*this).print();

            for (size_t row_idx = (size_t)(rows_-2); row_idx > (size_t)(row-1); row_idx--) {
                for (size_t col_idx = 0; col_idx < cols_; col_idx++) {
                    //std::cout << "rank: " << rank << " row_idx: " << row_idx << "\n";    
                    //std::cout << "rank: " << rank << " (*this).rows(): " << (*this).rows() << "\n";                
                    (*this)((size_t)(row_idx+1), col_idx) = (*this)((size_t)(row_idx), col_idx);
                }
            }
        }
        else {
            std::cout << "ERROR: Attempted to add row out of bounds for the matrix - row " << row << " on rank " << rank << "\n";
            exit(0);
        }
    }

    /*! \brief Remove row from col
     * Data are copied such that the first n elements in each row remain the same before and after this operation
     * \param [in] new_col      Desired number of columns in the enlarged matrix
     * \param [in] n_keep       Number of elements to preserve in all rows of the matrix
     */
    void remove_col(size_t col, int rank) {
        std::cout << "cols_: " << cols_ << " col: " << col << "\n";

        if ((col < cols_) && (col >= 0)) {
            for (size_t row_idx = 0; row_idx < rows_; row_idx++) {
                for (size_t col_idx = 0; col_idx < col; col_idx++) {
                    (*this)(row_idx, col_idx) = (*this)(row_idx, col_idx);
                }
            }
            for (size_t row_idx = 0; row_idx < rows_; row_idx++) {
                for (size_t col_idx = (size_t)(col+1); col_idx < cols_; col_idx++) {
                    (*this)(row_idx, (size_t)(col_idx-1)) = (*this)(row_idx, col_idx);
                }
            }
            reshape(rows_, (cols_ - 1));
        }
        else {
            std::cout << "ERROR: Attempted to remove col out of bounds for the matrix - col " << col << " on rank " << rank << "\n";
            exit(0);
        }
    }
        
    /*! \brief Increase number of columns in the matrix
     * Data are copied such that the first n[i] elements in each row remain the same before and after this operation
     * \param [in] new_col      Desired number of columns in the enlarged matrix
     * \param [in] n_keep       Number of elements to preserve in each row of the matrix (should have \p rows_ elements)
     */
    void enlarge_cols(size_t new_col, int *n_keep) {
        if (new_col > cols_) {
            size_t old_cols = cols_;
            reshape(rows_, new_col);
            
            size_t row_idx;
            for (row_idx = rows_; row_idx > 0; row_idx--) {
                auto begin = data_.begin();
                std::copy_backward(begin + (row_idx - 1) * old_cols, begin + (row_idx - 1) * old_cols + n_keep[row_idx - 1], begin + (row_idx - 1) * new_col + n_keep[row_idx - 1]);
            }
        }
    }
    
    /*! \brief Increase number of columns in the matrix
     * Data are copied such that the first n elements in each row remain the same before and after this operation
     * \param [in] new_col      Desired number of columns in the enlarged matrix
     * \param [in] n_keep       Number of elements to preserve in all rows of the matrix
     */
    void enlarge_cols(size_t new_col, int n_keep) {
        if (new_col > cols_) {
            size_t old_cols = cols_;
            reshape(rows_, new_col);
            
            size_t row_idx;
            for (row_idx = rows_; row_idx > 0; row_idx--) {
                auto begin = data_.begin();
                std::copy_backward(begin + (row_idx - 1) * old_cols, begin + (row_idx - 1) * old_cols + n_keep, begin + (row_idx - 1) * new_col + n_keep);
            }
        }
    }

    
    /*! \brief Change the dimensions without moving any of the data
     * \param [in] new_rows     Desired number of rows in the reshaped matrix
     * \param [in] new_cols     Desired number of columns in the reshaped matrix
     */
    void reshape(size_t new_rows, size_t new_cols, int rank=-1) {
        size_t new_size = new_rows * new_cols;
        
        if (new_size > tot_size_) {
            tot_size_ = new_size;
            data_.resize(tot_size_);
        }
            
        rows_ = new_rows;
        cols_ = new_cols;
    }
    
    /*! \return Current number of rows in matrix */
    size_t rows() const {
        return rows_;
    }

    /*! \return Current number of columns in matrix*/
    size_t cols() const {
        return cols_;
    }
    
    /*! \return Pointer to the data in the matrix*/
    mat_type *data() const {
        return (mat_type *) data_.data();
    }
    
    /*!
    * \brief Copies data from another matrix.
    * \param mat Matrix to copy from.
    */
    void copy_from(Matrix<mat_type> &mat) {
        std::copy(mat.data_.begin(), mat.data_.end(), data_.begin());
    }

    /*!
    * \brief Prints the matrix to standard output.
    */
    void print() {
        std::cout << "[ ";
        for (int m=0; m<(int)rows_; m++) {
            std::cout << "[ ";
            for (int n=0; n<(int)cols_; n++) {
                std::cout << (*this)((size_t)m, (size_t)n)  << " ";
            }
            std::cout << "] ";
            std::cout << "\n  ";
        }
        std::cout << "] \n\n";
    }
    
private:
    size_t rows_; /*!< Number of rows in the matrix */
    size_t cols_; /*!< Number of columns in the matrix */
    size_t tot_size_; /*!< Total size of the matrix */
    std::vector<mat_type> data_; /*!< Internal storage for matrix data */
};


/*! \brief A class for storing 4-D arrays of ints*/
class FourDArr {
public:
    /*! \brief Constructor
     * \param [in] len1 len2 len3 len4    Lengths of the 4 dimensions of the array
     */

    std::vector<size_t> size_vec;

    FourDArr(size_t len1, size_t len2, size_t len3, size_t len4)
    : len1_(len1)
    , len2_(len2)
    , len3_(len3)
    , len4_(len4)
    {
        data_ = (int *)malloc(sizeof(int) * len1 * len2 * len3 * len4);
        size_vec.push_back(len1);
        size_vec.push_back(len2);
        size_vec.push_back(len3);
        size_vec.push_back(len4);
    }
    
    /*! \brief Access an element of the 4-D array
     * \param [in] i1 First index
     * \param [in] i2 Second index
     * \param [in] i3 Third index
     * \param [in] i4 Fourth index
     * \returns Reference to array element
     */
    int& operator() (size_t i1, size_t i2, size_t i3, size_t i4) {
        return data_[i1 * len2_ * len3_ * len4_ + i2 * len3_ * len4_ + i3 * len4_ + i4];
    }
    
    int  operator() (size_t i1, size_t i2, size_t i3, size_t i4) const {
        return data_[i1 * len2_ * len3_ * len4_ + i2 * len3_ * len4_ + i3 * len4_ + i4];
    }

    /*! \brief Destructor*/
    ~FourDArr() {
        free(data_);
    }
    
    FourDArr(const FourDArr& m) = delete;
    
    FourDArr& operator= (const FourDArr& m) = delete;
    
    /*! \returns pointer to 0th element in the array */
    int *data() {
        return data_;
    }
    /*
    retrieve nonzero elements
    */
    Matrix<int> nonzero() {
        int vec_size = (int)(len1_*len2_*len3_*len4_)/8;
        std::cout << "nonzero \n"; 
        Matrix<int> mat_out(vec_size, 4);

        int elem = 0; 
        std::cout << "len1_: " << len1_ << " len2_: " << len2_ << " len3_: " << len3_ << " len4_: " << len4_ << "\n"; 
        
        for (int i1=0; i1<(int)len1_; i1++) {
            for (int i2=0; i2<(int)len2_; i2++) {
                for (int i3=0; i3<(int)len3_; i3++) {
                    for (int i4=0; i4<(int)len4_; i4++) {
                        if ( (*this)((size_t)i1, (size_t)i2, (size_t)i3, (size_t)i4) != false ) {
                            
                            if (elem >= (vec_size-1)) {
                                mat_out.reshape(vec_size*2, 4);
                                vec_size = vec_size * 2;
                            }
                            
                            mat_out[elem][0] = i1;
                            mat_out[elem][1] = i2;
                            mat_out[elem][2] = i3;
                            mat_out[elem][3] = i4;
                            elem ++;
                        }
                    }
                }
            }
        } 
        mat_out.reshape(elem, 4);
        return mat_out;
    }

    /*
    print data field of a FourDArr object
    */
    void print_4Dvector() { 
        std::cout << "\n[ ";
        for (int i1=0; i1<(int)len1_; i1++) {
            std::cout << "[ ";
            for (int i2=0; i2<(int)len2_; i2++) {
                std::cout << "[ ";
                for (int i3=0; i3<(int)len3_; i3++) {
                    std::cout << "[ ";
                    for (int i4=0; i4<(int)len4_; i4++) {
                        std::cout << (*this)((size_t)i1, (size_t)i2, (size_t)i3, (size_t)i4)  << " ";
                    }
                    std::cout << "] ";
                    std::cout << "\n  ";
                }
                std::cout << "] ";
                std::cout << "\n  ";
            }
            std::cout << "] ";
            std::cout << "\n  ";
        } 
        std::cout << "] \n\n";
    }

    /*
    retrieve elements to a 1D slice of a FourDArr object corresponding to coordinates input
    */
    std::vector<int> grab_idxs(std::vector< std::vector<int> > coords) {
        std::vector<int> output;

        for (int i=0; i<(int)coords.size(); i++) {
            std::vector<int> coord = coords[i];
            int w = coord[0];
            int x = coord[1];
            int y = coord[2];
            int z = coord[3];

            int value = (int)(*this)(w,x,y,z);
            output.push_back(value);
        }
        return output;
    }

    /*
    assign elements to a 1D slice of a FourDArr object corresponding to values input
    */
    void assign_idxs(std::vector< std::vector<int> > coords, std::vector<int> values) {
        size_t w;
        size_t x;
        size_t y;
        size_t z;
        
        

        for (int i=0; i<(int)coords.size(); i++) {
            std::vector<int> coord = coords[i];
            w = (size_t)coord[0];
            x = (size_t)coord[1];
            y = (size_t)coord[2];
            z = (size_t)coord[3];
            (*this)(w,x,y,z) = values[i];
        }
    }


    Matrix<int> nonzero_elems() {
        int vec_size = (int)(len1_*len2_*len3_*len4_)/8;
        Matrix<int> mat_out(vec_size, 5);
        mat_out.zero();

        int elem = 0; 
        for (int i1=0; i1<(int)len1_; i1++) {
            for (int i2=0; i2<(int)len2_; i2++) {
                for (int i3=0; i3<(int)len3_; i3++) {
                    for (int i4=0; i4<(int)len4_; i4++) {
                        if ( (*this)((size_t)i1, (size_t)i2, (size_t)i3, (size_t)i4) != 0 ) {
                           
                            if (elem >= (vec_size-1)) {
                                mat_out.reshape(vec_size*2, 5);
                                vec_size = vec_size * 2;
                            }
                            
                            mat_out[elem][0] = i1;
                            mat_out[elem][1] = i2;
                            mat_out[elem][2] = i3;
                            mat_out[elem][3] = i4;
                            mat_out[elem][4] = (*this)((size_t)i1, (size_t)i2, (size_t)i3, (size_t)i4);
                            elem ++;
                        }
                    }
                }
            }
        } 
        mat_out.reshape(elem, 5);
        return mat_out;
    }
    
    /*
    set all elements in data field to zero
    */
    void zero() {
        for (int i1=0; i1<len1_; i1++) {
            for (int i2=0; i2<len2_; i2++) {
                for (int i3=0; i3<len3_; i3++) {
                    for (int i4=0; i4<len4_; i4++) {
                        (*this)((size_t)i1, (size_t)i2, (size_t)i3, (size_t)i4) = 0;
                    }
                }
            }    
        }
    }

    void check_elems() {

        //std::cout << "check_elems()\n";
        for (int i1=0; i1<len1_; i1++) {
            for (int i2=0; i2<len2_; i2++) {
                for (int i3=0; i3<len3_; i3++) {
                    for (int i4=0; i4<len4_; i4++) {
                        if ((*this)((size_t)i1, (size_t)i2, (size_t)i3, (size_t)i4) != 0) {
                            //td::cout << "i1: " << i1 << " i2: " << i2 << " i3: " << i3 << " i4: " << i4 << " elem: " << (*this)((size_t)i1, (size_t)i2, (size_t)i3, (size_t)i4) << "\n";
                        }
                    }
                }
            }    
        }
    }

private:
    size_t len1_, len2_, len3_, len4_; ///< Dimensions of the array
    int* data_; ///< The data stored in the array
};

/*!
 * \class FourDBoolArr
 * \brief Represents a 4D boolean array implemented with bit-packing for memory efficiency.
 */
class FourDBoolArr {
    size_t len1_, len2_, len3_, len4_;
    std::vector<uint8_t> data_;

    /*!
     * \class BoolReference
     * \brief A helper class to manage individual bits in the 4D array.
     */
    class BoolReference {
    private:
        uint8_t &value_;
        uint8_t mask_;

        /*! \brief Sets the referenced bit to 0. */
        void zero() noexcept { value_ &= ~mask_; }

        /*! \brief Sets the referenced bit to 1. */
        void one() noexcept { value_ |= mask_; }

        /*! \brief Retrieves the current value of the referenced bit. */
        bool get() const noexcept { return !!(value_ & mask_); }

        /*! 
         * \brief Sets the referenced bit to the specified value.
         * \param b The boolean value to set.
         */
        void set(bool b) noexcept {
            if (b)
                one();
            else
                zero();
        }

    public:
        /*!
         * \brief Constructor for BoolReference.
         * \param value The byte containing the bit.
         * \param nbit The position of the bit within the byte.
         */
        BoolReference(uint8_t &value, uint8_t nbit)
            : value_(value), mask_(uint8_t(0x1) << nbit) {}

        /*! \brief Copy constructor for BoolReference. */
        BoolReference(const BoolReference &ref) : value_(ref.value_), mask_(ref.mask_) {}

        /*! \brief Assigns a boolean value to the referenced bit. */
        BoolReference &operator=(bool b) noexcept {
            set(b);
            return *this;
        }

        /*! \brief Copy assignment operator. */
        BoolReference &operator=(const BoolReference &br) noexcept { return *this = bool(br); }

        /*! \brief Implicit conversion to boolean. */
        operator bool() const noexcept { return get(); }
    };

public:
    std::vector<size_t> size_vec;

    /*!
     * \brief Constructor for FourDBoolArr.
     * \param len1 Length of the first dimension.
     * \param len2 Length of the second dimension.
     * \param len3 Length of the third dimension.
     * \param len4 Length of the fourth dimension.
     */
    FourDBoolArr(size_t len1, size_t len2, size_t len3, size_t len4)
        : len1_(len1), len2_(len2), len3_(len3), len4_(len4),
          data_(CEILING(len1 * len2 * len3 * len4, 8)) {
        size_vec.push_back(len1);
        size_vec.push_back(len2);
        size_vec.push_back(len3);
        size_vec.push_back(len4);
    }

    /*!
     * \brief Accesses an element of the 4D array.
     * \param i1 Index along the first dimension.
     * \param i2 Index along the second dimension.
     * \param i3 Index along the third dimension.
     * \param i4 Index along the fourth dimension.
     * \return A BoolReference to the specified bit.
     */
    BoolReference operator()(size_t i1, size_t i2, size_t i3, size_t i4) {
        size_t flat_idx = i1 * len2_ * len3_ * len4_ + i2 * len3_ * len4_ + i3 * len4_ + i4;
        size_t coarse_idx = flat_idx / 8;
        size_t fine_idx = flat_idx % 8;
        return BoolReference(data_[coarse_idx], fine_idx);
    }

    /*!
     * \brief Prints the elements of the 4D array to the console.
     */
    void print() { 
        std::cout << "\n[ ";
        for (int i1=0; i1<(int)len1_; i1++) {
            std::cout << "[ ";
            for (int i2=0; i2<(int)len2_; i2++) {
                std::cout << "[ ";
                for (int i3=0; i3<(int)len3_; i3++) {
                    std::cout << "[ ";
                    for (int i4=0; i4<(int)len4_; i4++) {
                        std::cout << (*this)((size_t)i1, (size_t)i2, (size_t)i3, (size_t)i4)  << " ";
                    }
                    std::cout << "] ";
                    std::cout << "\n  ";
                }
                std::cout << "] ";
                std::cout << "\n  ";
            }
            std::cout << "] ";
            std::cout << "\n  ";
        } 
        std::cout << "] \n\n";
    }
    
    /*!
     * \brief Retrieves the indices of non-zero elements.
     * \return A Matrix<int> containing the indices of non-zero elements.
     */
    Matrix<int> nonzero() {
        //std::cout << "nonzero enter \n";
        int vec_size = (int)(len1_*len2_*len3_*len4_)/8;
        Matrix<int> mat_out(vec_size, 4);

        int elem = 0; 
        
        for (int i1=0; i1<(int)len1_; i1++) {
            for (int i2=0; i2<(int)len2_; i2++) {
                for (int i3=0; i3<(int)len3_; i3++) {
                    for (int i4=0; i4<(int)len4_; i4++) {
                        if ( (*this)((size_t)i1, (size_t)i2, (size_t)i3, (size_t)i4) != false ) {
                            
                            if (elem >= (vec_size-2)) {

                                mat_out.reshape(vec_size*2, 4);
                                vec_size = vec_size * 2;
                            }
                            
                            mat_out[elem][0] = i1;
                            mat_out[elem][1] = i2;
                            mat_out[elem][2] = i3;
                            mat_out[elem][3] = i4;
                            elem ++;
                        }
                    }
                }
            }
        } 
        
        mat_out.reshape(elem, 4);
        return mat_out;
    }

    /*!
     * \brief Retrieves the indices and values of non-zero elements.
     * \return A Matrix<int> containing indices and values of non-zero elements.
     */
    Matrix<int> nonzero_elems() {
        int vec_size = (int)(len1_*len2_*len3_*len4_)/8;
        Matrix<int> mat_out(vec_size, 5);

        int elem = 0; 
        
        for (int i1=0; i1<(int)len1_; i1++) {
            for (int i2=0; i2<(int)len2_; i2++) {
                for (int i3=0; i3<(int)len3_; i3++) {
                    for (int i4=0; i4<(int)len4_; i4++) {
                        if ( (*this)((size_t)i1, (size_t)i2, (size_t)i3, (size_t)i4) != 0 ) {
                            
                            if (elem >= (vec_size-1)) {
                                mat_out.reshape(vec_size*2, 5);
                                vec_size = vec_size * 2;
                            }
                            
                            mat_out[elem][0] = i1;
                            mat_out[elem][1] = i2;
                            mat_out[elem][2] = i3;
                            mat_out[elem][3] = i4;
                            mat_out[elem][4] = (*this)((size_t)i1, (size_t)i2, (size_t)i3, (size_t)i4);
                            elem ++;
                        }
                    }
                }
            }
        } 
        mat_out.reshape(elem, 5);
        return mat_out;
    }

    /*!
     * \brief Sets all elements of the array to zero.
     */
    void zero() {
        std::fill(data_.begin(), data_.end(), 0);
    }

};

/*! \brief A class for storing 4-D arrays of ints*/
class FourDDoubleArr {
public:
    /*! \brief Constructor
     * \param [in] len1 len2 len3 len4    Lengths of the 4 dimensions of the array
     */

    std::vector<size_t> size_vec;
    std::vector<double> data_; ///< The data stored in the array

    FourDDoubleArr(size_t len1, size_t len2, size_t len3, size_t len4)
    : len1_(len1)
    , len2_(len2)
    , len3_(len3)
    , len4_(len4)
    , data_(len1 * len2 * len3 * len4)
    {
        //data_ = (double *)malloc(sizeof(double) * len1 * len2 * len3 * len4);
        size_vec.push_back(len1);
        size_vec.push_back(len2);
        size_vec.push_back(len3);
        size_vec.push_back(len4);
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
    
    double  operator() (size_t i1, size_t i2, size_t i3, size_t i4) const {
        return data_[i1 * len2_ * len3_ * len4_ + i2 * len3_ * len4_ + i3 * len4_ + i4];
    }
    
    /*
    retrieve nonzero elements
    */
    std::vector< std::vector<int> > nonzero() {
        int vec_size = (len1_*len2_*len3_*len4_)/8;
        std::vector<std::vector<int>> vec_out(vec_size, std::vector<int>(4));

        int elem = 0; 
        for (int i1=0; i1<(int)len1_; i1++) {
            for (int i2=0; i2<(int)len2_; i2++) {
                for (int i3=0; i3<(int)len3_; i3++) {
                    for (int i4=0; i4<(int)len4_; i4++) {
                        if ( (*this)((size_t)i1, (size_t)i2, (size_t)i3, (size_t)i4) != false ) {
                            if (elem >= (vec_size-1)) {
                                vec_out.resize(vec_size*2, std::vector<int>(4));
                                vec_size = vec_size * 2;
                            }
                            vec_out[elem][0] = i1;
                            vec_out[elem][1] = i2;
                            vec_out[elem][2] = i3;
                            vec_out[elem][3] = i4;
                            elem ++;
                        }
                    }
                }
            }
        } 
        vec_out.resize(elem);
        
        return vec_out;
    }

    /*
    print data field of a FourDArr object
    */
    void print_4Dvector() { 
        std::cout << "\n[ ";
        for (int i1=0; i1<(int)len1_; i1++) {
            std::cout << "[ ";
            for (int i2=0; i2<(int)len2_; i2++) {
                std::cout << "[ ";
                for (int i3=0; i3<(int)len3_; i3++) {
                    std::cout << "[ ";
                    for (int i4=0; i4<(int)len4_; i4++) {
                        std::cout << (*this)((size_t)i1, (size_t)i2, (size_t)i3, (size_t)i4)  << " ";
                    }
                    std::cout << "] ";
                    std::cout << "\n  ";
                }
                std::cout << "] ";
                std::cout << "\n  ";
            }
            std::cout << "] ";
            std::cout << "\n  ";
        } 
        std::cout << "] \n\n";
    }

    /*
    retrieve elements to a 1D slice of a FourDArr object corresponding to coordinates input
    */
    std::vector<int> grab_idxs(std::vector< std::vector<int> > coords) {
        std::vector<int> output;

        for (int i=0; i<(int)coords.size(); i++) {
            std::vector<int> coord = coords[i];
            int w = coord[0];
            int x = coord[1];
            int y = coord[2];
            int z = coord[3];

            int value = (int)(*this)(w,x,y,z);
            output.push_back(value);
        }
        return output;
    }

    /*
    assign elements to a 1D slice of a FourDArr object corresponding to values input
    */
    void assign_idxs(std::vector< std::vector<int> > coords, std::vector<int> values) {
        size_t w;
        size_t x;
        size_t y;
        size_t z;
        
        

        for (int i=0; i<(int)coords.size(); i++) {
            std::vector<int> coord = coords[i];
            w = (size_t)coord[0];
            x = (size_t)coord[1];
            y = (size_t)coord[2];
            z = (size_t)coord[3];
            (*this)(w,x,y,z) = values[i];
        }
    }

    /*
    return nonzero elements of 4Darray in form of matrix with indices 0-3 containing the coordinates
    and index 4 containing the element itself
    */
    Matrix<int> nonzero_elems() {
        int vec_size = (int)(len1_*len2_*len3_*len4_)/8;
        Matrix<int> mat_out(vec_size, 5);
        mat_out.zero();

        int elem = 0; 
        
        for (int i1=0; i1<(int)len1_; i1++) {
            for (int i2=0; i2<(int)len2_; i2++) {
                for (int i3=0; i3<(int)len3_; i3++) {
                    for (int i4=0; i4<(int)len4_; i4++) {
                        if ( (*this)((size_t)i1, (size_t)i2, (size_t)i3, (size_t)i4) != 0 ) {
                            
                            if (elem >= (vec_size-1)) {
                                mat_out.reshape(vec_size*2, 5);
                                vec_size = vec_size * 2;
                            }
                            
                            mat_out[elem][0] = i1;
                            mat_out[elem][1] = i2;
                            mat_out[elem][2] = i3;
                            mat_out[elem][3] = i4;
                            mat_out[elem][4] = (*this)((size_t)i1, (size_t)i2, (size_t)i3, (size_t)i4);
                            elem ++;
                        }
                    }
                }
            }
        } 
        mat_out.reshape(elem, 5);
        return mat_out;
    }
    
    /*
    set all elements in data field to zero
    */
    void zero() {
        for (int i1=0; i1<len1_; i1++) {
            for (int i2=0; i2<len2_; i2++) {
                for (int i3=0; i3<len3_; i3++) {
                    for (int i4=0; i4<len4_; i4++) {
                        (*this)((size_t)i1, (size_t)i2, (size_t)i3, (size_t)i4) = 0;
                    }
                }
            }    
        }
    }

private:
    size_t len1_, len2_, len3_, len4_; ///< Dimensions of the array

};

class SymmERIs {
    double *data_;
    size_t len_;
    
public:
    SymmERIs(size_t len) {
        size_t n_pair = len * (len + 1) / 2;
        len_ = len;
        size_t vec_size = n_pair * (n_pair + 1) / 2;
        data_ = (double *) malloc(vec_size * sizeof(double));
        std::fill(data_, data_ + vec_size, 0);
    }
    
    double chemist(size_t i1, size_t i2, size_t i3, size_t i4) const {
        size_t min1 = i1 < i2 ? i1 : i2;
        size_t max1 = i1 < i2 ? i2 : i1;
        size_t p1_idx = I_J_TO_TRI_WDIAG(min1, max1);
        size_t min2 = i3 < i4 ? i3 : i4;
        size_t max2 = i3 < i4 ? i4 : i3;
        size_t p2_idx = I_J_TO_TRI_WDIAG(min2, max2);
        size_t min_p = p1_idx < p2_idx ? p1_idx : p2_idx;
        size_t max_p = p1_idx < p2_idx ? p2_idx : p1_idx;
        return data_[I_J_TO_TRI_WDIAG(min_p, max_p)];
    }
    
    double &chemist_ordered(size_t i1, size_t i2, size_t i3, size_t i4) {
        size_t p1_idx = I_J_TO_TRI_WDIAG(i1, i2);
        size_t p2_idx = I_J_TO_TRI_WDIAG(i3, i4);
        return data_[I_J_TO_TRI_WDIAG(p1_idx, p2_idx)];
    }
    
    double physicist(size_t i1, size_t i2, size_t i3, size_t i4) const {
        return chemist(i1, i3, i2, i4);
    }
    
    ~SymmERIs() {
        free(data_);
    }
};

template<> class Matrix<bool>  {
    std::vector<uint8_t> data_;
    size_t rows_, cols_, tot_size_, cols_coarse_;
    
    class BoolReference
    {
    private:
        uint8_t & value_;
        uint8_t mask_;
        
        void zero(void) noexcept { value_ &= ~(mask_); }
        
        void one(void) noexcept { value_ |= (mask_); }
        
        bool get() const noexcept { return !!(value_ & mask_); }
        
        void set(bool b) noexcept
        {
            if(b)
                one();
            else
                zero();
        }
        
    public:
        BoolReference(uint8_t & value, uint8_t nbit)
        : value_(value), mask_(uint8_t(0x1) << nbit)
        { }
        
        BoolReference & operator=(bool b) noexcept { set(b); return *this; }
        
        BoolReference & operator=(const BoolReference & br) noexcept { return *this = bool(br); }
        
        operator bool() const noexcept { return get(); }
        };
        
    public:
        Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols), cols_coarse_(CEILING(cols, 8)), tot_size_(rows * CEILING(cols, 8)), data_(rows * CEILING(cols, 8), 0) {
        }
        
        BoolReference operator() (size_t row, size_t col) {
            BoolReference ref(data_[cols_coarse_ * row + col / 8], col % 8);
            return ref;
        }
        
        /*! \brief Change the dimensions without moving any of the data
         * \param [in] new_rows     Desired number of rows in the reshaped matrix
         * \param [in] new_cols     Desired number of columns in the reshaped matrix
         */
        void reshape(size_t new_rows, size_t new_cols) {
            cols_coarse_ = CEILING(new_cols, 8);
            size_t new_size = new_rows * cols_coarse_;
            if (new_size > tot_size_) {
                tot_size_ = new_size;
                data_.resize(tot_size_, 0);
            }
            rows_ = new_rows;
            cols_ = new_cols;
        }
        
        /*! \return Current number of columns in matrix*/
        size_t cols() const {
            return cols_;
        }
        
        class RowReference {
            private:
            size_t row_idx_;
            Matrix<bool> *mat_;
            Matrix<uint8_t> *other_mat_;
            
            public:
            RowReference(Matrix<bool> &mat, size_t row) : row_idx_(row), mat_(&mat), other_mat_(nullptr) {}
            
            RowReference(Matrix<uint8_t> *mat, size_t row) : row_idx_(row), other_mat_(mat), mat_(nullptr) {}
            
            BoolReference operator[] (size_t idx) {
                if (mat_) {
                    BoolReference ref(mat_->row_ptr(row_idx_)[idx / 8], idx % 8);
                    return ref;
                }
                else {
                    BoolReference ref((*other_mat_)(row_idx_, idx / 8), idx % 8);
                    return ref;
                }
            }
        };
        
        /*! \brief Access matrix row
         * \param [in] row      Row index
         * \return pointer to 0th element in a row of a matrix
         */
        RowReference operator[] (size_t row) {
            RowReference ref(*this, row);
            return ref;
        }
        
        /*! \brief Access matrix row
         * \param [in] row      Row index
         * \return pointer to 0th element in a row of a matrix
         */
        uint8_t *row_ptr (size_t row) const {
            return (uint8_t *)&data_[cols_coarse_ * row];
        }

        /*! \return Pointer to the  data in the matrix*/
        uint8_t *data() const {
            return (uint8_t *)data_.data();
        }
    };
    
#endif /* ndarr_h */
