/*! \file
 *
 * \brief Utilities for storing and manipulating sparse vectors
 *
 * Supports sparse vectors distributed among multiple processes if USE_MPI is
 * defined
 */

#include "vec_utils.h"


double vec_dot(dist_vec *vec, long long *idx2, double *vals2, size_t num2,
               unsigned long long *hashes2) {
    size_t hf_idx;
    ssize_t *ht_ptr;
    double numer = 0;
    for (hf_idx = 0; hf_idx < num2; hf_idx++) {
        ht_ptr = read_ht(vec->vec_hash, idx2[hf_idx], hashes2[hf_idx], 0);
        if (ht_ptr) {
            if (vec->type == INT) {
                int *int_vals = vec->values;
                numer += vals2[hf_idx] * int_vals[*ht_ptr];
            }
            else if (vec->type == DOUB) {
                double *doub_vals = vec->values;
                numer += vals2[hf_idx] * doub_vals[*ht_ptr];
            }
            else {
                fprintf(stderr, "Error: data type %d not supported in function vec_dot\n", vec->type);
            }
        }
    }
    return numer;
}


adder *init_adder(size_t size, dtype type) {
    adder *add_str = malloc(sizeof(adder));
    int n_procs = 1;
#ifdef USE_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
#endif
    add_str->send_idx = malloc(sizeof(long long) * size * n_procs);
    add_str->recv_idx = malloc(sizeof(long long) * size * n_procs);
    size_t el_size;
    if (type == INT) {
        el_size = sizeof(int);
    }
    else if (type == DOUB) {
        el_size = sizeof(double);
    }
    else {
        fprintf(stderr, "Error: data type %d not supported in init_adder\n", type);
        return NULL;
    }
    add_str->send_vals = malloc(el_size * size * n_procs);
    add_str->recv_vals = malloc(el_size * size * n_procs);
    add_str->send_cts = malloc(sizeof(int) * n_procs);
    add_str->recv_cts = malloc(sizeof(int) * n_procs);
    add_str->displacements = malloc(sizeof(int) * n_procs);
    add_str->size = size;
    add_str->type = type;
    size_t proc_idx;
    for (proc_idx = 0; proc_idx < n_procs; proc_idx++) {
        add_str->displacements[proc_idx] = (int)(proc_idx * size);
        add_str->send_cts[proc_idx] = 0;
    }
    return add_str;
}


void expand_adder(adder *ad_obj) {
    printf("Increasing storage capacity in adder\n");
    int n_procs = 1;
#ifdef USE_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
#endif
    size_t new_size = ad_obj->size * 2;
    size_t el_size;
    if (ad_obj->type == INT) {
        el_size = sizeof(int);
    }
    else if (ad_obj->type == DOUB) {
        el_size = sizeof(double);
    }
    else {
        fprintf(stderr, "Error: data type %d not supported in expand_adder\n", ad_obj->type);
        return;
    }
    ad_obj->send_vals = realloc(ad_obj->send_vals, new_size * el_size * n_procs);
    ad_obj->recv_vals = realloc(ad_obj->recv_vals, new_size * el_size * n_procs);
    ad_obj->send_idx = realloc(ad_obj->send_idx, sizeof(long long) * new_size * n_procs);
    ad_obj->recv_idx = realloc(ad_obj->recv_idx, sizeof(long long) * new_size * n_procs);
    ad_obj->size = new_size;
}


dist_vec *init_vec(size_t size, size_t add_size, mt_struct *rn_ptr, unsigned int n_orb,
                   unsigned int n_elec, dtype vec_type, int n_sites) {
    dist_vec *rt_ptr = malloc(sizeof(dist_vec));
    rt_ptr->indices = malloc(sizeof(long long) * size);
    rt_ptr->type = vec_type;
    if (vec_type == INT) {
        rt_ptr->values = malloc(sizeof(int) * size);
    }
    else if (vec_type == DOUB) {
        rt_ptr->values = malloc(sizeof(double) * size);
    }
    else {
        fprintf(stderr, "Error: type %d not supported in function init_vec\n", vec_type);
        return NULL;
    }
    rt_ptr->matr_el = malloc(sizeof(double) * size);
    rt_ptr->max_size = size;
    rt_ptr->curr_size = 0;
    rt_ptr->vec_hash = setup_ht(size, rn_ptr, 2 * n_orb);
    rt_ptr->vec_stack = setup_stack(1000);
    rt_ptr->tabl = gen_byte_table();
    rt_ptr->n_elec = n_elec;
    rt_ptr->occ_orbs = malloc(sizeof(unsigned char) * size * n_elec);
    rt_ptr->my_adder = init_adder(add_size, vec_type);
    rt_ptr->n_nonz = 0;
    rt_ptr->n_sites = n_sites;
    if (n_sites) {
        rt_ptr->neighb = malloc(sizeof(unsigned char) * size * 2 * (n_elec + 1));
    }
    else {
        rt_ptr->neighb = NULL;
    }
    return rt_ptr;
}


int idx_to_proc(dist_vec *vec, long long idx) {
    unsigned char orbs[vec->n_elec];
    gen_orb_list(idx, vec->tabl, orbs);
    unsigned long long hash_val = hash_fxn(orbs, vec->n_elec, vec->proc_scrambler);
    int n_procs = 1;
#ifdef USE_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
#endif
    return hash_val % n_procs;
}

unsigned long long idx_to_hash(dist_vec *vec, long long idx) {
    unsigned char orbs[vec->n_elec];
    gen_orb_list(idx, vec->tabl, orbs);
    return hash_fxn(orbs, vec->n_elec, vec->vec_hash->scrambler);
}

// Double size of buffers used for storage in vector
void expand_vector(dist_vec *vec) {
    printf("Increasing storage capacity in vector\n");
    size_t new_max = vec->max_size * 2;
    vec->indices = realloc(vec->indices, sizeof(long long) * new_max);
    vec->matr_el = realloc(vec->matr_el, sizeof(double) * new_max);
    vec->occ_orbs = realloc(vec->occ_orbs, sizeof(unsigned char) * new_max * vec->n_elec);
    if (vec->neighb) {
        vec->neighb = realloc(vec->neighb, sizeof(unsigned char) * new_max * 2 * (vec->n_elec + 1));
    }
    
    size_t el_size = 0;
    if (vec->type == INT) {
        el_size = sizeof(int);
    }
    else if (vec->type == DOUB) {
        el_size = sizeof(double);
    }
    vec->values = realloc(vec->values, el_size * new_max);
}


void add_int(dist_vec *vec, long long idx, int val, long long ini_flag) {
    if (val == 0) {
        return;
    }
    if (vec->type != INT) {
        fprintf(stderr, "Error: cannot add integer to a vector that is not of integer type.\n");
    }
    int proc_idx = idx_to_proc(vec, idx);
    adder *tmp_adder = vec->my_adder;
    long long (*idx_2D)[tmp_adder->size] = (long long (*)[tmp_adder->size])tmp_adder->send_idx;
    int (*val_2D)[tmp_adder->size] = (int (*)[tmp_adder->size])tmp_adder->send_vals;
    int *count = &(tmp_adder->send_cts[proc_idx]);
    if (*count == tmp_adder->size) {
        expand_adder(tmp_adder);
    }
    idx_2D[proc_idx][*count] = idx | ini_flag;
    val_2D[proc_idx][*count] = val;
    (*count)++;
}


void add_doub(dist_vec *vec, long long idx, double val, long long ini_flag) {
    if (fabs(val) < 1e-10) {
        return;
    }
    if (vec->type != DOUB) {
        fprintf(stderr, "Error: cannot add double to a vector that is not of double type.\n");
    }
    int proc_idx = idx_to_proc(vec, idx);
    adder *tmp_adder = vec->my_adder;
    long long (*idx_2D)[tmp_adder->size] = (long long (*)[tmp_adder->size])tmp_adder->send_idx;
    double (*val_2D)[tmp_adder->size] = (double (*)[tmp_adder->size])tmp_adder->send_vals;
    int *count = &(tmp_adder->send_cts[proc_idx]);
    if (*count == tmp_adder->size) {
        expand_adder(tmp_adder);
    }
    idx_2D[proc_idx][*count] = idx | ini_flag;
    val_2D[proc_idx][*count] = val;
    (*count)++;
}


void perform_add(dist_vec *vec, long long ini_bit) {
    adder *addr = vec->my_adder;
    int n_procs = 1;
    int proc_idx;
    long long (*idx_r_2D)[addr->size] = (long long (*)[addr->size])addr->recv_idx;
    
    size_t el_size = 0;
    if (vec->type == INT) {
        el_size = sizeof(int);
    }
    else if (vec->type == DOUB) {
        el_size = sizeof(double);
    }
    else {
        fprintf(stderr, "Error: type %d not supported in function perform_add.\n", vec->type);
    }
    char (*recv_buf)[addr->size * el_size] = (char (*)[addr->size * el_size])addr->recv_vals;
#ifdef USE_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Alltoall(addr->send_cts, 1, MPI_INT, addr->recv_cts, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Alltoallv(addr->send_idx, addr->send_cts, addr->displacements, MPI_LONG_LONG, addr->recv_idx, addr->recv_cts, addr->displacements, MPI_LONG_LONG, MPI_COMM_WORLD);
    if (vec->type == INT) {
        MPI_Alltoallv(addr->send_vals, addr->send_cts, addr->displacements, MPI_INT, addr->recv_vals, addr->recv_cts, addr->displacements, MPI_INT, MPI_COMM_WORLD);
    }
    else if (vec->type == DOUB) {
        MPI_Alltoallv(addr->send_vals, addr->send_cts, addr->displacements, MPI_DOUBLE, addr->recv_vals, addr->recv_cts, addr->displacements, MPI_DOUBLE, MPI_COMM_WORLD);
    }
    else {
        fprintf(stderr, "Error: type %d not supported in function perform_add.\n", vec->type);
    }
#else
    char (*send_buf)[addr->size * el_size] = (char (*)[addr->size * el_size])addr->send_vals;
    long long (*idx_s_2D)[addr->size] = (long long (*)[addr->size])addr->send_idx;
    for (proc_idx = 0; proc_idx < n_procs; proc_idx++) {
        int cpy_size = addr->send_cts[proc_idx];
        addr->recv_cts[proc_idx] = cpy_size;
        memcpy(idx_r_2D[proc_idx], idx_s_2D[proc_idx], cpy_size * sizeof(long long));
        memcpy(recv_buf[proc_idx], send_buf[proc_idx], cpy_size * el_size);
    }
#endif
    // Move elements from receiving buffers to vector
    for (proc_idx = 0; proc_idx < n_procs; proc_idx++) {
        addr->send_cts[proc_idx] = 0;
        size_t el_idx;
        for (el_idx = 0; el_idx < addr->recv_cts[proc_idx]; el_idx++) {
            long long new_idx = idx_r_2D[proc_idx][el_idx];
            int ini_flag = !(!(new_idx & ini_bit));
            new_idx &= ini_bit - 1;
            unsigned long long hash_val = idx_to_hash(vec, new_idx);
            ssize_t *idx_ptr = read_ht(vec->vec_hash, new_idx, hash_val, ini_flag);
            if (idx_ptr && *idx_ptr == -1) {
                *idx_ptr = pop(vec->vec_stack);
                if (*idx_ptr == -1) {
                    if (vec->curr_size >= vec->max_size) {
                        expand_vector(vec);
                    }
                    *idx_ptr = vec->curr_size;
                    vec->curr_size++;
                }
                if (vec->type == INT) {
                    ((int *)vec->values)[*idx_ptr] = 0;
                }
                else if (vec->type == DOUB) {
                    ((double *)vec->values)[*idx_ptr] = 0;
                }
                gen_orb_list(new_idx, vec->tabl, &(vec->occ_orbs[vec->n_elec * *idx_ptr]));
                vec->indices[*idx_ptr] = new_idx;
                vec->matr_el[*idx_ptr] = NAN;
                vec->n_nonz++;
                if (vec->neighb) {
                    unsigned char (*tmp_neib)[vec->n_elec + 1] = (unsigned char (*)[vec->n_elec + 1]) &(vec->neighb[*idx_ptr * 2 * (vec->n_elec + 1)]);
                    find_neighbors_1D(new_idx, vec->n_sites, vec->tabl, vec->n_elec, tmp_neib);
                }
            }
            int delete = 0;
            if (vec->type == INT) {
                int *recv_ints = (int *)recv_buf[proc_idx];
                int *vec_vals = (int *)vec->values;
                if (ini_flag || (idx_ptr && (vec_vals[*idx_ptr] * recv_ints[el_idx]) > 0)) {
                    vec_vals[*idx_ptr] += recv_ints[el_idx];
                    delete = vec_vals[*idx_ptr] == 0;
                }
            }
            else if (vec->type == DOUB) {
                double *recv_doubs = (double *)recv_buf[proc_idx];
                double *vec_vals = (double *)vec->values;
                if (ini_flag || (idx_ptr && (vec_vals[*idx_ptr] * recv_doubs[el_idx]) > 0)) {
                    vec_vals[*idx_ptr] += recv_doubs[el_idx];
                    delete = vec_vals[*idx_ptr] == 0;
                }
            }
            if (delete == 1) {
                push(vec->vec_stack, *idx_ptr);
                del_ht(vec->vec_hash, new_idx, hash_val);
                vec->n_nonz--;
            }
        }
    }
}

void del_at_pos(dist_vec *vec, size_t pos) {
    long long idx = vec->indices[pos];
    unsigned long long hash_val = idx_to_hash(vec, idx);
    push(vec->vec_stack, pos);
    del_ht(vec->vec_hash, idx, hash_val);
    vec->n_nonz--;
}


int *int_at_pos(dist_vec *vec, size_t pos) {
    int *vals = (int *)vec->values;
    return &vals[pos];
}


double *doub_at_pos(dist_vec *vec, size_t pos) {
    double *vals = (double *)vec->values;
    return &vals[pos];
}


unsigned char *orbs_at_pos(dist_vec *vec, size_t pos) {
    unsigned char (*orbs)[vec->n_elec] = (unsigned char (*)[vec->n_elec])vec->occ_orbs;
    return orbs[pos];
}


double local_norm(dist_vec *vec) {
    double norm = 0;
    if (vec->type == INT) {
        int *vals = (int *)vec->values;
        size_t idx;
        for (idx = 0; idx < vec->curr_size; idx++) {
            norm += abs(vals[idx]);
        }
    }
    else if (vec->type == DOUB) {
        double *vals = (double *)vec->values;
        size_t idx;
        for (idx = 0; idx < vec->curr_size; idx++) {
            norm += fabs(vals[idx]);
        }
    }
    return norm;
}

void save_vec(dist_vec *vec, const char *path) {
    int my_rank = 0;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
#endif
    
    size_t el_size = 0;
    if (vec->type == INT) {
        el_size = sizeof(int);
    }
    else if (vec->type == DOUB) {
        el_size = sizeof(double);
    }
    
    char buffer[100];
    sprintf(buffer, "%sdets%d.dat", path, my_rank);
    FILE *file_p = fopen(buffer, "wb");
    fwrite(vec->indices, sizeof(long long), vec->curr_size, file_p);
    fclose(file_p);
    
    sprintf(buffer, "%svals%d.dat", path, my_rank);
    file_p = fopen(buffer, "wb");
    fwrite(vec->values, el_size, vec->curr_size, file_p);
    fclose(file_p);
}


void load_vec(dist_vec *vec, const char *path) {
    int my_rank = 0;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
#endif
    
    size_t el_size = 0;
    if (vec->type == INT) {
        el_size = sizeof(int);
    }
    else if (vec->type == DOUB) {
        el_size = sizeof(double);
    }
    
    size_t n_dets;
    char buffer[100];
    sprintf(buffer, "%sdets%d.dat", path, my_rank);
    FILE *file_p = fopen(buffer, "rb");
    n_dets = fread(vec->indices, sizeof(long long), 10000000, file_p);
    fclose(file_p);
    
    sprintf(buffer, "%svals%d.dat", path, my_rank);
    file_p = fopen(buffer, "rb");
    fread(vec->values, el_size, n_dets, file_p);
    fclose(file_p);
    
    size_t det_idx;
    int n_nonz = 0;
    for (det_idx = 0; det_idx < n_dets; det_idx++) {
        int is_nonz = 0;
        if (vec->type == DOUB) {
            double *doub_vals = vec->values;
            if (fabs(doub_vals[det_idx]) > 1e-9) {
                is_nonz = 1;
                doub_vals[n_nonz] = doub_vals[det_idx];
            }
        }
        else if (vec->type == INT) {
            int *int_vals = vec->values;
            if (int_vals[det_idx] != 0) {
                is_nonz = 1;
                int_vals[n_nonz] = int_vals[det_idx];
            }
        }
        if (is_nonz) {
            gen_orb_list(vec->indices[det_idx], vec->tabl, &(vec->occ_orbs[vec->n_elec * n_nonz]));
            vec->indices[n_nonz] = vec->indices[det_idx];
            vec->matr_el[n_nonz] = NAN;
            n_nonz++;
            if (vec->neighb) {
                unsigned char (*tmp_neib)[vec->n_elec + 1] = (unsigned char (*)[vec->n_elec + 1]) &(vec->neighb[det_idx * 2 * (vec->n_elec + 1)]);
                find_neighbors_1D(vec->indices[det_idx], vec->n_sites, vec->tabl, vec->n_elec, tmp_neib);
            }
        }
    }
    vec->n_nonz = n_nonz;
    vec->curr_size = n_nonz;
}


unsigned char gen_orb_list(long long det, byte_table *table, unsigned char *occ_orbs) {
    unsigned int byte_idx, elec_idx;
    long long mask = 255;
    unsigned char n_elec, det_byte, bit_idx;
    elec_idx = 0;
    byte_idx = 0;
    unsigned char tot_elec = 0;
    while (det != 0) {
        det_byte = det & mask;
        n_elec = table->nums[det_byte];
        for (bit_idx = 0; bit_idx < n_elec; bit_idx++) {
            occ_orbs[elec_idx + bit_idx] = (8 * byte_idx + table->pos[det_byte][bit_idx]);
        }
        elec_idx = elec_idx + n_elec;
        det = det >> 8;
        byte_idx = byte_idx + 1;
        tot_elec += n_elec;
    }
    return tot_elec;
}
