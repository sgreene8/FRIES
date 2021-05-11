/*! \file
 *
 * \brief Miscellaneous math utilities and definitions
 */

#include "math_utils.h"


unsigned int bits_between(uint8_t *bit_str, uint8_t a, uint8_t b) {
    unsigned int n_bits = 0;
    uint8_t byte_counts[] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};
    uint8_t min_bit, max_bit;
    
    if (a < b) {
        min_bit = a;
        max_bit = b;
    }
    else {
        max_bit = a;
        min_bit = b;
    }
    
    uint8_t mask;
    uint8_t min_byte = min_bit / 8;
    uint8_t max_byte = max_bit / 8;
    int same_byte = min_byte == max_byte;
    uint8_t byte_idx;
    uint8_t curr_int;
    
    byte_idx = min_bit / 8;
    min_bit %= 8;
    max_bit %= 8;
    if (same_byte) {
        mask = (1 << max_bit) - (1 << (min_bit + 1));
    }
    else {
        mask = 256 - (1 << (min_bit + 1));
    }
    curr_int = mask & bit_str[byte_idx];
    n_bits = byte_counts[curr_int & 15];
    curr_int >>= 4;
    n_bits += byte_counts[curr_int & 15];
    for (byte_idx++; byte_idx < max_byte; byte_idx++) {
        curr_int = bit_str[byte_idx];
        n_bits += byte_counts[curr_int & 15];
        curr_int >>= 4;
        n_bits += byte_counts[curr_int & 15];
    }
    if (!same_byte) {
        mask = (1 << max_bit) - 1;
        curr_int = mask & bit_str[byte_idx];
        n_bits += byte_counts[curr_int & 15];
        curr_int >>= 4;
        n_bits += byte_counts[curr_int & 15];
    }
    
    return n_bits;
}

uint8_t find_bits(const uint8_t *bit_str, uint8_t *bits, uint8_t n_bytes) {
    uint8_t n_bits = 0;
    uint8_t byte_idx;
    __m128i bit_offset_v = _mm_set_epi8(8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0);
    __m128i sixteen = _mm_set1_epi8(16);
    for (byte_idx = 0; byte_idx < n_bytes - 1; byte_idx += 2) {
        uint8_t byte1 = bit_str[byte_idx];
        uint8_t byte1_nbits = _mm_popcnt_u32(byte1);
        
        uint8_t byte2 = bit_str[byte_idx + 1];
        uint8_t byte2_nbits = _mm_popcnt_u32(byte2);
        
        __m128i bit_vec = _mm_set_epi64x(byte_pos[byte2], byte_pos[byte1]);
        bit_vec += bit_offset_v;
        uint8_t *vec_ptr = (uint8_t *)&bit_vec;
        
        memcpy(bits + n_bits, vec_ptr, byte1_nbits);
        n_bits += byte1_nbits;
        memcpy(bits + n_bits, vec_ptr + 8, byte2_nbits);
        n_bits += byte2_nbits;
        
        bit_offset_v += sixteen;
    }
    if (byte_idx < n_bytes) {
        uint8_t byte = bit_str[byte_idx];
        uint8_t byte_nbits = _mm_popcnt_u32(byte);

        __m128i bit_vec = _mm_set_epi64x(0, byte_pos[byte]);
        bit_vec += bit_offset_v;
        uint8_t *vec_ptr = (uint8_t *)&bit_vec;

        memcpy(bits + n_bits, vec_ptr, byte_nbits);
        n_bits += byte_nbits;
    }
    
    return n_bits;
}

uint8_t find_diff_bits(const uint8_t *str1, const uint8_t *str2, uint8_t *bits, uint8_t n_bytes) {
    uint8_t n_bits = 0;
    uint8_t byte_idx;
    __m128i bit_offset_v = _mm_set_epi8(8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0);
    __m128i sixteen = _mm_set1_epi8(16);
    for (byte_idx = 0; byte_idx < n_bytes - 1; byte_idx += 2) {
        uint8_t byte1 = str1[byte_idx] ^ str2[byte_idx];
        uint8_t byte1_nbits = _mm_popcnt_u32(byte1);
        
        uint8_t byte2 = str1[byte_idx + 1] ^ str2[byte_idx + 1];
        uint8_t byte2_nbits = _mm_popcnt_u32(byte2);
        
        if (n_bits + byte1_nbits + byte2_nbits > 4) {
            return UINT8_MAX;
        }
        
        __m128i bit_vec = _mm_set_epi64x(byte_pos[byte2], byte_pos[byte1]);
        bit_vec += bit_offset_v;
        uint8_t *vec_ptr = (uint8_t *)&bit_vec;
        
        memcpy(bits + n_bits, vec_ptr, byte1_nbits);
        n_bits += byte1_nbits;
        memcpy(bits + n_bits, vec_ptr + 8, byte2_nbits);
        n_bits += byte2_nbits;
        
        bit_offset_v += sixteen;
    }
    if (byte_idx < n_bytes) {
        uint8_t byte = str1[byte_idx] ^ str2[byte_idx];
        uint8_t byte_nbits = _mm_popcnt_u32(byte);
        
        if (n_bits + byte_nbits > 4) {
            return UINT8_MAX;
        }

        __m128i bit_vec = _mm_set_epi64x(0, byte_pos[byte]);
        bit_vec += bit_offset_v;
        uint8_t *vec_ptr = (uint8_t *)&bit_vec;

        memcpy(bits + n_bits, vec_ptr, byte_nbits);
        n_bits += byte_nbits;
    }
    
    return n_bits;
}


void new_sorted(uint8_t *restrict orig_list, uint8_t *restrict new_list,
                uint8_t length, uint8_t del_idx, uint8_t new_el) {
    uint8_t offset = 0;
    uint8_t idx;
    if (new_el > orig_list[del_idx]) {
        memcpy(new_list, orig_list, sizeof(uint8_t) * del_idx);
        for (idx = del_idx + 1; idx < length; idx++) {
            if (orig_list[idx] < new_el) {
                offset++;
            }
        }
        memcpy(new_list + del_idx, orig_list + del_idx + 1, sizeof(uint8_t) * offset);
        new_list[del_idx + offset] = new_el;
        memcpy(new_list + del_idx + offset + 1, orig_list + del_idx + offset + 1, sizeof(uint8_t) * (length - del_idx - offset - 1));
    }
    else {
        for (idx = 0; idx < del_idx; idx++) {
            if (orig_list[idx] > new_el) {
                offset++;
            }
        }
        memcpy(new_list, orig_list, sizeof(uint8_t) * (del_idx - offset));
        new_list[del_idx - offset] = new_el;
        memcpy(new_list + del_idx - offset + 1, orig_list + del_idx - offset, sizeof(uint8_t) * offset);
        memcpy(new_list + del_idx + 1, orig_list + del_idx + 1, (length - del_idx - 1) * sizeof(uint8_t));
    }
}

void repl_sorted(uint8_t *srt_list, uint8_t length, uint8_t del_idx, uint8_t new_el) {
    uint8_t offset = 0;
    uint8_t idx;
    if (new_el > srt_list[del_idx]) {
        for (idx = del_idx + 1; idx < length; idx++) {
            if (srt_list[idx] < new_el) {
                offset++;
            }
        }
        memmove(srt_list + del_idx, srt_list + del_idx + 1, offset * sizeof(uint8_t));
        srt_list[del_idx + offset] = new_el;
    }
    else {
        for (idx = 0; idx < del_idx; idx++) {
            if (srt_list[idx] > new_el) {
                offset++;
            }
        }
        memmove(srt_list + del_idx - offset + 1, srt_list + del_idx - offset, offset * sizeof(uint8_t));
        srt_list[del_idx - offset] = new_el;
    }
}
