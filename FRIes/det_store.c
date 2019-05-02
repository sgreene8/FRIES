//
//  det_store.c
//  FRIes
//
//  Created by Samuel Greene on 4/14/19.
//  Copyright © 2019 Samuel Greene. All rights reserved.
//

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
        size_t *new_arr = malloc(sizeof(size_t) * 2 * curr_size);
        buf->buf_size = 2 * curr_size;
        size_t j;
        for (j = 0; j < curr_size; j++) {
            new_arr[j] = buf->storage[j];
        }
        free(buf->storage);
        buf->storage = new_arr;
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
    table->bucket_sizes = malloc(sizeof(unsigned int) * table_size);
    table->bucket_vals = malloc(sizeof(size_t *) * table_size);
    table->bucket_dets = malloc(sizeof(long long *) * table_size);
    table->scrambler = malloc(sizeof(unsigned int) * rn_len);
    
    size_t j;
    for (j = 0; j < table_size; j++) {
        table->bucket_dets[j] = NULL;
    }
    for (j = 0; j < rn_len; j++) {
        table->scrambler[j] = genrand_mt(rn_gen);
    }
    return table;
}


ssize_t *read_ht(hash_table *table, long long det, unsigned long long hash_val,
                 int create) {
    size_t table_idx = hash_val % table->length;
    
    if (table->bucket_dets[table_idx] == NULL) {
        table->bucket_dets[table_idx] = malloc(sizeof(long long) * MaxBuckets);
        table->bucket_vals[table_idx] = malloc(sizeof(ssize_t) * MaxBuckets);
        table->bucket_sizes[table_idx] = 0;
    }
    
    long long *det_array = table->bucket_dets[table_idx];
    unsigned int *num_ptr = &table->bucket_sizes[table_idx];
    unsigned int search_idx = 0;
    ssize_t *ret_ptr = NULL;
    
    for (search_idx = 0; search_idx < *num_ptr; search_idx++) {
        if (det_array[search_idx] == det) {
            ret_ptr = &(table->bucket_vals[table_idx][search_idx]);
            break;
        }
    }
    if (search_idx == *num_ptr && create) { // not found, so add it
        if (*num_ptr == MaxBuckets) {
            fprintf(stderr, "out of space in a row of the hash table; too many hash collisions\n");
            FILE *bucket_f = fopen("buckets.txt", "w");
            for (table_idx = 0; table_idx < table->length; table_idx++) {
                fprintf(bucket_f, "%u\n", table->bucket_sizes[table_idx]);
            }
            fclose(bucket_f);
        }
        else {
            det_array[search_idx] = det;
            (*num_ptr)++;
            ret_ptr = &(table->bucket_vals[table_idx][search_idx]);
            *ret_ptr = -1;
        }
    }
    return ret_ptr;
}

void del_ht(hash_table *table, long long det, unsigned long long hash_val) {
    size_t row_idx = hash_val % table->length;
    long long *det_array = table->bucket_dets[row_idx];
    ssize_t *val_array = table->bucket_vals[row_idx];
    unsigned int *num_ptr = &table->bucket_sizes[row_idx];
    unsigned int search_idx = 0;
    
    for (search_idx = 0; search_idx < (*num_ptr); search_idx++) {
        if (det_array[search_idx] == det) {
            det_array[search_idx] = det_array[(*num_ptr) - 1];
            val_array[search_idx] = val_array[(*num_ptr) - 1];
            (*num_ptr)--;
            break;
        }
    }
}
