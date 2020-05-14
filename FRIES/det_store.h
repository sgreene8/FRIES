/*! \file
 *
 * \brief Utilities for keeping track of Slater determinant indices of a
 * sparse vector.
 */

#ifndef det_store_h
#define det_store_h

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <FRIES/Ext_Libs/dcmt/dc.h>
#include <FRIES/math_utils.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Test bit string equality
 *
 * \param [in] str1     First bit string to compare
 * \param [in] str2     Second bit string to compare
 * \param [in] n_bytes      Number of bytes represented in \p str1 and \p str2
 * \return 1 if str1 == str2, 0 otherwise
 */
int bit_str_equ(uint8_t *str1, uint8_t *str2, uint8_t n_bytes);


/*! \brief Read the nth bit from a bit string
 * \param [in] bit_str      The bit string to read from
 * \param [in] bit_idx      The index of the bit to read
 * \return The value of the bit (0 or 1)
 */
inline int read_bit(const uint8_t *bit_str, uint8_t bit_idx) {
    uint8_t byte_idx = bit_idx / 8;
    return !(!(bit_str[byte_idx] & (1 << (bit_idx % 8))));
}


/*! \brief Set the nth bit of a bit string to 0
 * \param [in] bit_str      The bit string to set
 * \param [in] bit_idx      The index of the bit to set
 */
void zero_bit(uint8_t *bit_str, uint8_t bit_idx);


/*! \brief Set the nth bit of a bit string to 1
 * \param [in] bit_str      The bit string to set
 * \param [in] bit_idx      The index of the bit to set
 */
void set_bit(uint8_t *bit_str, uint8_t bit_idx);


/*! \brief Convert a binary bit string to a text string for printing
 * \param [in] bit_str      The binary bit string to print
 * \param [in] n_bytes      The number of bytes in the bit string
 * \param [out] out_str     The buffer that contains the text string upon return
 */
void print_str(uint8_t *bit_str, uint8_t n_bytes, char *out_str);


#ifdef __cplusplus
}
#endif

#endif /* det_store_h */
