/*! \file 
 *
 * \brief Miscellaneous math utilities and definitions
 */

#ifndef math_utils_h
#define math_utils_h

#include <stdlib.h>

/*! \brief A table used to identify positions of 1's in binary representation of
 *  a number
 */
typedef struct {
    unsigned char (*pos)[8]; ///< 2-D array whose rows contain lists of positions of 1's in binary representation of row index (dimensions 256 x 8)
    unsigned char *nums; ///< 1-D array containing number of 1's in binary representation of index (length 256)
} byte_table;


typedef enum {
    DOUB,
    INT
} dtype;


/*! \brief Generate and initialize a byte_table struct
 *
 * Calculates a lookup table used to decompose a byte into a list of positions
 * of 1's (for 0-255)
 *
 * \return byte_table structure initialized with information about numbers 0-255
 */
byte_table *gen_byte_table(void);


/*! \brief Count the number of 1's in the binary representation of a number
 *
 * \param [in] num      Number to be analyzed
 * \param [in] tab      Pointer to a byte_table struct
 */
unsigned int count_bits(long long num, byte_table *tab);


/*! \brief Count number of 1's between two bits in binary representation of a
 * number
 *
 * The order of the two bit indices does not matter
 *
 * \param [in] bit_str  The number whose bits will be counted
 * \param [in] a        The position of one of the bits in question
 * \param [in] b        The position of the second bit in question
 * \return the number of bits
 */
unsigned int bits_between(long long bit_str, unsigned char a, unsigned char b);


#endif /* math_utils_h */
