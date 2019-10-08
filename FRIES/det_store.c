/*! \file
 *
 * \brief Utilities for keeping track of Slater determinant indices of a
 * sparse vector.
 */

#include "det_store.h"

unsigned long long hash_fxn(unsigned char *occ_orbs, unsigned int n_elec, unsigned int *rand_nums) {
    unsigned long long hash = 0;
    unsigned int i;
    for (i = 0; i < n_elec; i++) {
        hash = 1099511628211LL * hash + (i + 1) * rand_nums[occ_orbs[i]];
    }
    return hash;
}


void push(stack_s *buf, size_t val) {
    size_t idx = buf->curr_idx;
    size_t curr_size = buf->buf_size;
    if (idx == curr_size) {
        // Reallocate storage
        printf("Doubling size of stack\n");
        buf->storage = realloc(buf->storage, 2 * curr_size);
        buf->buf_size = 2 * curr_size;
    }
    buf->storage[idx] = val;
    buf->curr_idx = idx + 1;
}


ssize_t pop(stack_s *buf) {
    size_t idx = buf->curr_idx;
    if (idx == 0) {
        return -1;
    }
    idx--;
    buf->curr_idx = idx;
    return buf->storage[idx];
}


stack_s *setup_stack(size_t buf_size) {
    stack_s *ret_stack = malloc(sizeof(stack_s));
    ret_stack->buf_size = buf_size;
    ret_stack->curr_idx = 0;
    ret_stack->storage = malloc(sizeof(size_t) * buf_size);
    return ret_stack;
}


hash_table *setup_ht(size_t table_size, mt_struct *rn_gen, unsigned int rn_len) {
    hash_table *table = malloc(sizeof(hash_table));
    table->length = table_size;
    table->buckets = malloc(sizeof(hash_entry *) * table_size);
    table->scrambler = malloc(sizeof(unsigned int) * rn_len);
    table->recycle_list = NULL;
    
    size_t j;
    for (j = 0; j < table_size; j++) {
        table->buckets[j] = NULL;
    }
    for (j = 0; j < rn_len; j++) {
        table->scrambler[j] = genrand_mt(rn_gen);
    }
    return table;
}


ssize_t *read_ht(hash_table *table, long long det, unsigned long long hash_val,
                 int create) {
    size_t table_idx = hash_val % table->length;
    // address of location storing address of next entry
    hash_entry **prev_ptr = &(table->buckets[table_idx]);
    // address of next entry
    hash_entry *next_ptr = *prev_ptr;
    unsigned int collisions = 0;
    while (next_ptr) {
        if (next_ptr->det == det) {
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
        }
        *prev_ptr = next_ptr;
        next_ptr->det = det;
        next_ptr->next = NULL;
        next_ptr->val = -1;
        return &(next_ptr->val);
    }
    else
        return NULL;
}

void del_ht(hash_table *table, long long det, unsigned long long hash_val) {
    size_t table_idx = hash_val % table->length;
    // address of location storing address of next entry
    hash_entry **prev_ptr = &(table->buckets[table_idx]);
    // address of next entry
    hash_entry *next_ptr = *prev_ptr;
    while (next_ptr) {
        if (next_ptr->det == det) {
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
