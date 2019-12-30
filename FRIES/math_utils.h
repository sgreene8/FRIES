/*! \file 
 *
 * \brief Miscellaneous math utilities and definitions
 */

#ifndef math_utils_h
#define math_utils_h

#include <stdlib.h>
#include <stdint.h>

#define CEILING(x,y) ((x + y - 1) / y)

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief A table used to identify positions of 1's in binary representation of
 *  a number
 */
typedef struct {
    uint8_t (*pos)[8]; ///< 2-D array whose rows contain lists of positions of 1's in binary representation of row index (dimensions 256 x 8)
    uint8_t *nums; ///< 1-D array containing number of 1's in binary representation of index (length 256)
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


/*! \brief Generate a list of the indices of the nonzero bits in a bit string
 *
 * \param [in] bit_str      The bit string to decode
 * \param [out] bits        The array in which to store the bit indices
 * \param [in] n_bytes      The length of \p bit_str
 * \param [in] tabl     A byte_table struct to use to decode the bits
 */
uint8_t find_bits(uint8_t *bit_str, uint8_t *bits, uint8_t n_bytes, byte_table *tabl);

/*! \brief Count number of 1's between two bits in binary representation of a
 * number
 *
 * The order of the two bit indices does not matter
 *
 * \param [in] bit_str  The binary representation of the number in bit string format
 * \param [in] a        The position of one of the bits in question
 * \param [in] b        The position of the second bit in question
 * \return the number of bits
 */
unsigned int bits_between(uint8_t *bit_str, uint8_t a, uint8_t b);


#ifdef __cplusplus
}
#endif

#endif /* math_utils_h */
