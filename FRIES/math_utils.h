/*! \file 
 *
 * \brief Miscellaneous math utilities and definitions
 */

#ifndef math_utils_h
#define math_utils_h

#include <stdlib.h>
#include <stdint.h>
#include <string.h>

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

//const uint8_t byte_nums[256] = {0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,};


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
uint8_t find_bits(const uint8_t *bit_str, uint8_t *bits, uint8_t n_bytes, const byte_table *tabl);

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


/*! \brief Given an ordered list of numbers, return a copy, modified such that the element at a specified index is
 * replaced with a new element and re-sorted
 *
 * \param [in] orig_list     The original ordered list of numbers
 * \param [out] new_list    The resulting, new list
 * \param [in] length   The number of elements in the original and new lists
 * \param [in]  del_idx     The index of the item to be replaced
 * \param [in]  new_el      The element replacing the removed element
 */
void new_sorted(uint8_t *orig_list, uint8_t *new_list,
                uint8_t length, uint8_t del_idx, uint8_t new_el);


/*! \brief Given an ordered list of numbers, replace an element at a specified index with a new element and re-sort
 *
 * \param [in] srt_list     The original ordered list of numbers
 * \param [in] length   The number of elements in the original and new lists
 * \param [in]  del_idx     The index of the item to be replaced
 * \param [in]  new_el      The element replacing the removed element
 */
 void repl_sorted(uint8_t *srt_list, uint8_t length, uint8_t del_idx, uint8_t new_el);


#ifdef __cplusplus
}
#endif

#endif /* math_utils_h */
