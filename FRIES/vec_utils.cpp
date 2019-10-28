/*! \file
 *
 * \brief Utilities for storing and manipulating sparse vectors
 *
 * Supports sparse vectors distributed among multiple processes if USE_MPI is
 * defined
 */

#include "vec_utils.hpp"


//template <class el_type>
//double DistVec<el_type>::dot(long long *idx2, double *vals2, size_t num2,
//           unsigned long long *hashes2)


//adder *init_adder(size_t size, dtype type) {
//    adder *add_str = (adder *)malloc(sizeof(adder));
//    int n_procs = 1;
//#ifdef USE_MPI
//    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
//#endif
//    add_str->send_idx = (long long *)malloc(sizeof(long long) * size * n_procs);
//    add_str->recv_idx = (long long *)malloc(sizeof(long long) * size * n_procs);
//    size_t el_size;
//    if (type == INT) {
//        el_size = sizeof(int);
//    }
//    else if (type == DOUB) {
//        el_size = sizeof(double);
//    }
//    else {
//        fprintf(stderr, "Error: data type %d not supported in init_adder\n", type);
//        return NULL;
//    }
//    add_str->send_vals = malloc(el_size * size * n_procs);
//    add_str->recv_vals = malloc(el_size * size * n_procs);
//    add_str->send_cts = (int *)malloc(sizeof(int) * n_procs);
//    add_str->recv_cts = (int *)malloc(sizeof(int) * n_procs);
//    add_str->displacements = (int *)malloc(sizeof(int) * n_procs);
//    add_str->size = size;
//    add_str->type = type;
//    size_t proc_idx;
//    for (proc_idx = 0; proc_idx < n_procs; proc_idx++) {
//        add_str->displacements[proc_idx] = (int)(proc_idx * size);
//        add_str->send_cts[proc_idx] = 0;
//    }
//    return add_str;
//}

//template <class el_type>
//void DistVec<el_type>::expand()

//void expand_adder(adder *ad_obj) {
//    printf("Increasing storage capacity in adder\n");
//    int n_procs = 1;
//#ifdef USE_MPI
//    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
//#endif
//    size_t new_size = ad_obj->size * 2;
//    size_t el_size;
//    if (ad_obj->type == INT) {
//        el_size = sizeof(int);
//    }
//    else if (ad_obj->type == DOUB) {
//        el_size = sizeof(double);
//    }
//    else {
//        fprintf(stderr, "Error: data type %d not supported in expand_adder\n", ad_obj->type);
//        return;
//    }
//    ad_obj->send_vals = realloc(ad_obj->send_vals, new_size * el_size * n_procs);
//    ad_obj->recv_vals = realloc(ad_obj->recv_vals, new_size * el_size * n_procs);
//    ad_obj->send_idx = (long long *)realloc(ad_obj->send_idx, sizeof(long long) * new_size * n_procs);
//    ad_obj->recv_idx = (long long *)realloc(ad_obj->recv_idx, sizeof(long long) * new_size * n_procs);
//    ad_obj->size = new_size;
//}


//dist_vec *init_vec(size_t size, size_t add_size, mt_struct *rn_ptr, unsigned int n_orb,
//                   unsigned int n_elec, dtype vec_type, int n_sites) {
//    dist_vec *rt_ptr = (dist_vec *)malloc(sizeof(dist_vec));
//    rt_ptr->indices = (long long *)malloc(sizeof(long long) * size);
//    rt_ptr->type = vec_type;
//    if (vec_type == INT) {
//        rt_ptr->values = malloc(sizeof(int) * size);
//    }
//    else if (vec_type == DOUB) {
//        rt_ptr->values = malloc(sizeof(double) * size);
//    }
//    else {
//        fprintf(stderr, "Error: type %d not supported in function init_vec\n", vec_type);
//        return NULL;
//    }
//    rt_ptr->matr_el = (double *)malloc(sizeof(double) * size);
//    rt_ptr->max_size = size;
//    rt_ptr->curr_size = 0;
//    rt_ptr->vec_hash = setup_ht(size, rn_ptr, 2 * n_orb);
//    rt_ptr->vec_stack = NULL; //setup_stack(1000);
//    rt_ptr->tabl = gen_byte_table();
//    rt_ptr->n_elec = n_elec;
////    rt_ptr->occ_orbs = (unsigned char *)malloc(sizeof(unsigned char) * size * n_elec);
//    rt_ptr->occ_orbs = new Matrix<unsigned char>(size, n_elec);
//    rt_ptr->my_adder = init_adder(add_size, vec_type);
//    rt_ptr->n_nonz = 0;
//    rt_ptr->n_sites = n_sites;
//    if (n_sites) {
////        rt_ptr->neighb = malloc(sizeof(unsigned char) * size * 2 * (n_elec + 1));
//        rt_ptr->neighb = new Matrix<unsigned char>(size, 2 * (n_elec + 1));
//    }
//    else {
//        rt_ptr->neighb = NULL;
//    }
//    return rt_ptr;
//}

//template <class el_type>
//unsigned long long DistVec<el_type>::idx_to_hash(long long idx)

//template <class el_type>
//void DistVec<el_type>::perform_add(long long ini_bit)

// Double size of buffers used for storage in vector
//void expand_vector(dist_vec *vec) {
//    printf("Increasing storage capacity in vector\n");
//    size_t new_max = vec->max_size * 2;
//    vec->indices = (long long *)realloc(vec->indices, sizeof(long long) * new_max);
//    vec->matr_el = (double *)realloc(vec->matr_el, sizeof(double) * new_max);
////    vec->occ_orbs = (unsigned char *)realloc(vec->occ_orbs, sizeof(unsigned char) * new_max * vec->n_elec);
//    vec->occ_orbs->enlarge(new_max);
//    if (vec->neighb) {
////        vec->neighb = realloc(vec->neighb, sizeof(unsigned char) * new_max * 2 * (vec->n_elec + 1));
//        vec->neighb->enlarge(new_max * 2);
//    }
//
//    size_t el_size = 0;
//    if (vec->type == INT) {
//        el_size = sizeof(int);
//    }
//    else if (vec->type == DOUB) {
//        el_size = sizeof(double);
//    }
//    vec->values = realloc(vec->values, el_size * new_max);
//}


//void add_int(dist_vec *vec, long long idx, int val, long long ini_flag) {
//    if (val == 0) {
//        return;
//    }
//    if (vec->type != INT) {
//        fprintf(stderr, "Error: cannot add integer to a vector that is not of integer type.\n");
//    }
//    int proc_idx = idx_to_proc(vec, idx);
//    adder *tmp_adder = vec->my_adder;
////    long long (*idx_2D)[tmp_adder->size] = (long long (*)[tmp_adder->size])tmp_adder->send_idx;
//    int *val_2D = (int *)tmp_adder->send_vals;
//    int *count = &(tmp_adder->send_cts[proc_idx]);
//    if (*count == tmp_adder->size) {
//        expand_adder(tmp_adder);
////        idx_2D = (long long (*)[tmp_adder->size])tmp_adder->send_idx;
////        val_2D = (int (*)[tmp_adder->size])tmp_adder->send_vals;
//        val_2D = (int *)tmp_adder->send_vals;
//    }
//    tmp_adder->send_idx->operator()(proc_idx, *count) = idx | ini_flag;
//    val_2D[proc_idx * tmp_adder->size + *count] = val;
//    (*count)++;
//}


//void add_doub(dist_vec *vec, long long idx, double val, long long ini_flag) {
//    if (fabs(val) < 1e-10) {
//        return;
//    }
//    if (vec->type != DOUB) {
//        fprintf(stderr, "Error: cannot add double to a vector that is not of double type.\n");
//    }
//    int proc_idx = idx_to_proc(vec, idx);
//    adder *tmp_adder = vec->my_adder;
////    long long (*idx_2D)[tmp_adder->size] = (long long (*)[tmp_adder->size])tmp_adder->send_idx;
//    double *val_2D = (double *)tmp_adder->send_vals;
//    int *count = &(tmp_adder->send_cts[proc_idx]);
//    if (*count == tmp_adder->size) {
//        expand_adder(tmp_adder);
////        idx_2D = (long long (*)[tmp_adder->size])tmp_adder->send_idx;
////        val_2D = (double (*)[tmp_adder->size])tmp_adder->send_vals;
//        val_2D = (double *)tmp_adder->send_vals;
//    }
//    tmp_adder->send_idx->operator()(proc_idx, *count) = idx | ini_flag;
//    val_2D[proc_idx * tmp_adder->size + *count] = val;
//    (*count)++;
//}


//void perform_add(dist_vec *vec, long long ini_bit) {
//    adder *addr = vec->my_adder;
//    int n_procs = 1;
//    int proc_idx;
//    long long (*idx_r_2D)[addr->size] = (long long (*)[addr->size])addr->recv_idx;
//
//    size_t el_size = 0;
//    if (vec->type == INT) {
//        el_size = sizeof(int);
//    }
//    else if (vec->type == DOUB) {
//        el_size = sizeof(double);
//    }
//    else {
//        fprintf(stderr, "Error: type %d not supported in function perform_add.\n", vec->type);
//    }
//    char (*recv_buf)[addr->size * el_size] = (char (*)[addr->size * el_size])addr->recv_vals;
//#ifdef USE_MPI
//    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
//    MPI_Alltoall(addr->send_cts, 1, MPI_INT, addr->recv_cts, 1, MPI_INT, MPI_COMM_WORLD);
//    MPI_Alltoallv(addr->send_idx, addr->send_cts, addr->displacements, MPI_LONG_LONG, addr->recv_idx, addr->recv_cts, addr->displacements, MPI_LONG_LONG, MPI_COMM_WORLD);
//    if (vec->type == INT) {
//        MPI_Alltoallv(addr->send_vals, addr->send_cts, addr->displacements, MPI_INT, addr->recv_vals, addr->recv_cts, addr->displacements, MPI_INT, MPI_COMM_WORLD);
//    }
//    else if (vec->type == DOUB) {
//        MPI_Alltoallv(addr->send_vals, addr->send_cts, addr->displacements, MPI_DOUBLE, addr->recv_vals, addr->recv_cts, addr->displacements, MPI_DOUBLE, MPI_COMM_WORLD);
//    }
//    else {
//        fprintf(stderr, "Error: type %d not supported in function perform_add.\n", vec->type);
//    }
//#else
//    char (*send_buf)[addr->size * el_size] = (char (*)[addr->size * el_size])addr->send_vals;
//    long long (*idx_s_2D)[addr->size] = (long long (*)[addr->size])addr->send_idx;
//    for (proc_idx = 0; proc_idx < n_procs; proc_idx++) {
//        int cpy_size = addr->send_cts[proc_idx];
//        addr->recv_cts[proc_idx] = cpy_size;
//        memcpy(idx_r_2D[proc_idx], idx_s_2D[proc_idx], cpy_size * sizeof(long long));
//        memcpy(recv_buf[proc_idx], send_buf[proc_idx], cpy_size * el_size);
//    }
//#endif
//    // Move elements from receiving buffers to vector
//    for (proc_idx = 0; proc_idx < n_procs; proc_idx++) {
//        addr->send_cts[proc_idx] = 0;
//        size_t el_idx;
//        for (el_idx = 0; el_idx < addr->recv_cts[proc_idx]; el_idx++) {
//            long long new_idx = idx_r_2D[proc_idx][el_idx];
//            int ini_flag = !(!(new_idx & ini_bit));
//            new_idx &= ini_bit - 1;
//            unsigned long long hash_val = idx_to_hash(vec, new_idx);
//            ssize_t *idx_ptr = read_ht(vec->vec_hash, new_idx, hash_val, ini_flag);
//            if (idx_ptr && *idx_ptr == -1) {
//                *idx_ptr = pop_stack(vec);
//                if (*idx_ptr == -1) {
//                    if (vec->curr_size >= vec->max_size) {
//                        expand_vector(vec);
//                    }
//                    *idx_ptr = vec->curr_size;
//                    vec->curr_size++;
//                }
//                if (vec->type == INT) {
//                    ((int *)vec->values)[*idx_ptr] = 0;
//                }
//                else if (vec->type == DOUB) {
//                    ((double *)vec->values)[*idx_ptr] = 0;
//                }
//                if (gen_orb_list(new_idx, vec->tabl, vec->occ_orbs->operator[](*idx_ptr)) != vec->n_elec) {
//                    fprintf(stderr, "Error: determinant %lld created with an incorrect number of electrons.\n", new_idx);
//                }
//                vec->indices[*idx_ptr] = new_idx;
//                vec->matr_el[*idx_ptr] = NAN;
//                vec->n_nonz++;
//                if (vec->neighb) {
//                    unsigned char *tmp_neib = vec->neighb->operator[](*idx_ptr * 2 * (vec->n_elec + 1));
//                    find_neighbors_1D(new_idx, vec->n_sites, vec->tabl, vec->n_elec, tmp_neib);
//                }
//            }
//            int del_bool = 0;
//            if (vec->type == INT) {
//                int *recv_ints = (int *)recv_buf[proc_idx];
//                int *vec_vals = (int *)vec->values;
//                if (ini_flag || (idx_ptr && (vec_vals[*idx_ptr] * recv_ints[el_idx]) > 0)) {
//                    vec_vals[*idx_ptr] += recv_ints[el_idx];
//                    del_bool = vec_vals[*idx_ptr] == 0;
//                }
//            }
//            else if (vec->type == DOUB) {
//                double *recv_doubs = (double *)recv_buf[proc_idx];
//                double *vec_vals = (double *)vec->values;
//                if (ini_flag || (idx_ptr && (vec_vals[*idx_ptr] * recv_doubs[el_idx]) > 0)) {
//                    vec_vals[*idx_ptr] += recv_doubs[el_idx];
//                    del_bool = vec_vals[*idx_ptr] == 0;
//                }
//            }
//            if (del_bool == 1) {
//                push_stack(vec, *idx_ptr);
//                del_ht(vec->vec_hash, new_idx, hash_val);
//                vec->n_nonz--;
//            }
//        }
//    }
//}

//template <class el_type>
//void DistVec<el_type>::del_at_pos(size_t pos)


//int *int_at_pos(dist_vec *vec, size_t pos) {
//    int *vals = (int *)vec->values;
//    return &vals[pos];
//}
//
//
//double *doub_at_pos(dist_vec *vec, size_t pos) {
//    double *vals = (double *)vec->values;
//    return &vals[pos];
//}

//template <class el_type>
//unsigned char * DistVec<el_type>::orbs_at_pos(size_t pos)

//template <class el_type>
//double DistVec<el_type>::local_norm()

//template <class el_type>
//void DistVec<el_type>::save(const char *path)


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


void find_neighbors_1D(long long det, unsigned int n_sites, byte_table *table,
                       unsigned int n_elec, unsigned char *neighbors) {
    long long neib_bits = det & ~(det >> 1);
    long long mask = (1LL << (2 * n_sites)) - 1;
    mask ^= (1LL << (n_sites - 1));
    mask ^= (1LL << (2 * n_sites - 1));
    neib_bits &= mask; // open boundary conditions
    neighbors[0] = gen_orb_list(neib_bits, table, &neighbors[1]);
    
    neib_bits = det & (~det << 1);
    mask = (1LL << (2 * n_sites)) - 1;
    mask ^= (1LL << n_sites);
    neib_bits &= mask; // open boundary conditions
    neighbors[n_elec + 1] = gen_orb_list(neib_bits, table, &neighbors[n_elec + 1 + 1]);
}

//template <class el_type>
//void DistVec<el_type>::push_stack(size_t idx)

//template <class el_type>
//ssize_t DistVec<el_type>::pop_stack()


//void DistVec<>::collect_procs()
