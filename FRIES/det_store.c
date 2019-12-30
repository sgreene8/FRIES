/*! \file
 *
 * \brief Utilities for keeping track of Slater determinant indices of a
 * sparse vector.
 */

#include "det_store.h"

uintmax_t hash_fxn(uint8_t *occ_orbs, unsigned int n_elec, uint8_t *phonon_nums, unsigned int n_phonon, unsigned int *rand_nums) {
    uintmax_t hash = 0;
    unsigned int i;
    for (i = 0; i < n_elec; i++) {
        hash = 1099511628211LL * hash + (i + 1) * rand_nums[occ_orbs[i]];
    }
    for (i = 0; i < n_phonon; i++) {
        hash = 1099511628211LL * hash + (i + 1) * rand_nums[phonon_nums[i]];
    }
    return hash;
}


hash_table *setup_ht(size_t table_size, mt_struct *rn_gen, unsigned int rn_len) {
    hash_table *table = malloc(sizeof(hash_table));
    table->length = table_size;
    table->buckets = malloc(sizeof(hash_entry *) * table_size);
    table->scrambler = malloc(sizeof(unsigned int) * rn_len);
    table->recycle_list = NULL;
    table->idx_size = CEILING(rn_len, 8);
    
    for (size_t j = 0; j < table_size; j++) {
        table->buckets[j] = NULL;
    }
    for (size_t j = 0; j < rn_len; j++) {
        table->scrambler[j] = genrand_mt(rn_gen);
    }
    return table;
}


ssize_t *read_ht(hash_table *table, uint8_t *det, uintmax_t hash_val,
                 int create) {
    size_t table_idx = hash_val % table->length;
    // address of location storing address of next entry
    hash_entry **prev_ptr = &(table->buckets[table_idx]);
    // address of next entry
    hash_entry *next_ptr = *prev_ptr;
    unsigned int collisions = 0;
    while (next_ptr) {
        if (bit_str_equ(det, next_ptr->det, table->idx_size)) {
            break;
        }
        collisions++;
        prev_ptr = &(next_ptr->next);
        next_ptr = *prev_ptr;
    }
    if (collisions > 20) {
        fprintf(stderr, "There is a line in the hash table with >20 hash collisions.\n");
    }
    if (next_ptr) {
        return &(next_ptr->val);
    }
    else if (create) {
        if (table->recycle_list) {
            next_ptr = table->recycle_list;
            table->recycle_list = next_ptr->next;
        }
        else {
            next_ptr = malloc(sizeof(hash_entry));
            next_ptr->det = malloc(table->idx_size);
        }
        *prev_ptr = next_ptr;
        memcpy(next_ptr->det, det, table->idx_size);
        next_ptr->next = NULL;
        next_ptr->val = -1;
        return &(next_ptr->val);
    }
    else
        return NULL;
}

void del_ht(hash_table *table, uint8_t *det, uintmax_t hash_val) {
    size_t table_idx = hash_val % table->length;
    // address of location storing address of next entry
    hash_entry **prev_ptr = &(table->buckets[table_idx]);
    // address of next entry
    hash_entry *next_ptr = *prev_ptr;
    while (next_ptr) {
        if (bit_str_equ(next_ptr->det, det, table->idx_size)) {
            break;
        }
        prev_ptr = &(next_ptr->next);
        next_ptr = *prev_ptr;
    }
    if (next_ptr) {
        *prev_ptr = next_ptr->next;
        next_ptr->next = table->recycle_list;
        table->recycle_list = next_ptr;
    }
}


int bit_str_equ(uint8_t *str1, uint8_t *str2, uint8_t n_bytes) {
    for (int byte_idx = 0; byte_idx < n_bytes; byte_idx++) {
        if (str1[byte_idx] != str2[byte_idx]) {
            return 0;
        }
    }
    return 1;
}


int read_bit(const uint8_t *bit_str, uint8_t bit_idx) {
    uint8_t byte_idx = bit_idx / 8;
    return !(!(bit_str[byte_idx] & (1 << (bit_idx % 8))));
}


void zero_bit(uint8_t *bit_str, uint8_t bit_idx) {
    uint8_t *byte = &bit_str[bit_idx / 8];
    uint8_t mask = ~0 ^ (1 << (bit_idx % 8));
    *byte &= mask;
}

void set_bit(uint8_t *bit_str, uint8_t bit_idx) {
    uint8_t *byte = &bit_str[bit_idx / 8];
    uint8_t mask = 1 << (bit_idx % 8);
    *byte |= mask;
}

void print_str(uint8_t *bit_str, uint8_t n_bytes, char *out_str) {
    for (uint8_t byte_idx = n_bytes; byte_idx > 0; byte_idx--) {
        sprintf(&out_str[(n_bytes - byte_idx) * 2], "%02x", bit_str[byte_idx - 1]);
    }
    out_str[n_bytes * 2] = '\0';
}
