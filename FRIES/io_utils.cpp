/*! \file
 *
 * \brief Utilities for reading/writing data from/to disk.
 */

#include "io_utils.hpp"

size_t read_csv(double *buf, const char *fname) {
    size_t row_idx = 0;
    size_t n_col;
    size_t n_read = 0;
    //                                   file, delimiter, first_line_is_header?
    CsvParser *csvparser = CsvParser_new(fname, ",", 0);
    CsvRow *row;
    
    while ((row = CsvParser_getRow(csvparser)) ) {
        const char **rowFields = CsvParser_getFields(row);
        n_col = CsvParser_getNumFields(row);
        n_read += n_col;
        for (int i = 0 ; i < n_col ; i++) {
            sscanf(rowFields[i], "%lf", &buf[n_col * row_idx + i]);
        }
        CsvParser_destroy_row(row);
        row_idx++;
    }
    CsvParser_destroy(csvparser);
    return n_read;
}

size_t read_csv(uint8_t *buf, const char *fname) {
    size_t row_idx = 0;
    size_t n_col;
    size_t n_read = 0;
    //                                   file, delimiter, first_line_is_header?
    CsvParser *csvparser = CsvParser_new(fname, " ", 0);
    CsvRow *row;
    unsigned int scan_val;
    
    while ((row = CsvParser_getRow(csvparser)) ) {
        const char **rowFields = CsvParser_getFields(row);
        n_col = CsvParser_getNumFields(row);
        n_read += n_col;
        for (int i = 0 ; i < n_col ; i++) {
            sscanf(rowFields[i], "%u", &scan_val);
            buf[n_col * row_idx + i] = scan_val;
        }
        CsvParser_destroy_row(row);
        row_idx++;
    }
    CsvParser_destroy(csvparser);
    return n_read;
}


size_t read_csv(int *buf, const char *fname) {
    size_t row_idx = 0;
    size_t n_col;
    size_t n_read = 0;
    
    CsvParser *csvparser = CsvParser_new(fname, " ", 0);
    CsvRow *row;
    int scan_val;
    
    while ((row = CsvParser_getRow(csvparser)) ) {
        const char **rowFields = CsvParser_getFields(row);
        n_col = CsvParser_getNumFields(row);
        n_read += n_col;
        for (int i = 0 ; i < n_col ; i++) {
            sscanf(rowFields[i], "%d", &scan_val);
            buf[n_col * row_idx + i] = scan_val;
        }
        CsvParser_destroy_row(row);
        row_idx++;
    }
    CsvParser_destroy(csvparser);
    return n_read;
}


void parse_hf_input(const std::string &hf_dir, hf_input *in_struct) {
    std::string path(hf_dir);
    path.append("sys_params.txt");
    std::ifstream in_file(path);
    if (!in_file.is_open()) {
        throw std::runtime_error("Could not open file sys_params.txt");
    }
    
    std::string keyword;
    std::getline(in_file, keyword);
    bool success = keyword == "n_elec";
    
    if (success) {
        in_file >> in_struct->n_elec;
    }
    else {
        throw std::runtime_error("Fould not find n_elec parameter in sys_params.txt");
    }
    
    unsigned int n_frz;
    std::getline(in_file, keyword);
    std::getline(in_file, keyword);
    success = keyword == "n_frozen";
    if (success) {
        in_file >> n_frz;
    }
    else {
        throw std::runtime_error("Fould not find n_frozen parameter in sys_params.txt");
    }
    in_struct->n_frz = n_frz;
    
    std::getline(in_file, keyword);
    std::getline(in_file, keyword);
    success = keyword == "n_orb";
    if (success) {
        in_file >> in_struct->n_orb;
    }
    else {
        throw std::runtime_error("Fould not find n_orb parameter in sys_params.txt");
    }
    
    std::getline(in_file, keyword);
    std::getline(in_file, keyword);
    success = keyword == "eps";
    if (success) {
        in_file >> in_struct->eps;
    }
    else {
        throw std::runtime_error("Fould not find eps parameter in sys_params.txt");
    }
    
    std::getline(in_file, keyword);
    std::getline(in_file, keyword);
    success = keyword == "hf_energy";
    if (success) {
        in_file >> in_struct->hf_en;
    }
    else {
        throw std::runtime_error("Fould not find hf_energy parameter in sys_params.txt");
    }
    
    in_file.close();
    
    unsigned int tot_orb = in_struct->n_orb + n_frz / 2;
    path = hf_dir;
    path.append("symm.txt");
    in_struct->symm = (uint8_t *)malloc(sizeof(uint8_t) * tot_orb);
    read_csv(in_struct->symm, path.c_str());
    in_struct->symm = &(in_struct->symm[n_frz / 2]);
    
    path = hf_dir;
    path.append("hcore.txt");
    in_struct->hcore = new Matrix<double>(tot_orb, tot_orb);
    size_t n_read = read_csv((*(in_struct->hcore)).data(), path.c_str());
    if (n_read < tot_orb * tot_orb) {
        std::stringstream msg;
        msg << "Could not read " << tot_orb * tot_orb << " elements from " << path;
        throw std::runtime_error(msg.str());
    }
    
    path = hf_dir;
    path.append("eris.txt");
    in_struct->eris = new FourDArr(tot_orb, tot_orb, tot_orb, tot_orb);
    n_read = read_csv(in_struct->eris->data(), path.c_str());
    if (n_read < tot_orb * tot_orb * tot_orb * tot_orb) {
        std::stringstream msg;
        msg << "Could not read " << tot_orb * tot_orb * tot_orb * tot_orb << " elements from " << path;
        throw std::runtime_error(msg.str());
    }
}

void parse_hh_input(const std::string &hh_path, hh_input *in_struct) {
    std::ifstream file_p(hh_path);
    if (!file_p.is_open()) {
        throw std::runtime_error("Could not open file containing Hubbard-Holstein parameters");
    }
    
    std::string keyword;
    std::getline(file_p, keyword);
    bool success = keyword == "n_elec";
    
    if (success) {
        file_p >> in_struct->n_elec;
    }
    else {
        throw std::runtime_error("Could not find n_elec parameter in file containing Hubbard-Holstein parameters");
    }
    
    std::getline(file_p, keyword);
    std::getline(file_p, keyword);
    success = keyword == "lat_len";
    if (success) {
        file_p >> in_struct->lat_len;
    }
    else {
        throw std::runtime_error("Could not find lat_len parameter in file containing Hubbard-Holstein parameters");
    }
    
    std::getline(file_p, keyword);
    std::getline(file_p, keyword);
    success = keyword == "n_dim";
    if (success) {
        file_p >> in_struct->n_dim;
    }
    else {
        throw std::runtime_error("Could not find n_dim parameter in file containing Hubbard-Holstein parameters");
    }
    
    std::getline(file_p, keyword);
    std::getline(file_p, keyword);
    success = keyword == "eps";
    if (success) {
        file_p >> in_struct->eps;
    }
    else {
        throw std::runtime_error("Could not find eps parameter in file containing Hubbard-Holstein parameters");
    }
    
    std::getline(file_p, keyword);
    std::getline(file_p, keyword);
    success = keyword == "U";
    if (success) {
        file_p >> in_struct->elec_int;
    }
    else {
        throw std::runtime_error("Could not find electron interaction parameter (U) in file containing Hubbard-Holstein parameters");
    }
    
    std::getline(file_p, keyword);
    std::getline(file_p, keyword);
    success = keyword == "omega";
    if (success) {
        file_p >> in_struct->ph_freq;
    }
    else {
        throw std::runtime_error("Could not find phonon frequency parameter (omega) in file containing Hubbard-Holstein parameters");
    }
    
    std::getline(file_p, keyword);
    std::getline(file_p, keyword);
    success = keyword == "g";
    if (success) {
        file_p >> in_struct->elec_ph;
    }
    else {
        throw std::runtime_error("Could not find electron-phonon interaction parameter (g) in file containing Hubbard-Holstein parameters");
    }
    
    std::getline(file_p, keyword);
    std::getline(file_p, keyword);
    success = keyword == "gs_energy";
    if (success) {
        file_p >> in_struct->hf_en;
    }
    else {
        throw std::runtime_error("Could not find gs_energy parameter in file containing Hubbard-Holstein parameters");
    }
    
    file_p.close();
}

size_t load_vec_txt(const std::string &prefix, Matrix<uint8_t> &dets, int *vals) {
        int my_rank = 0;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
#endif
    
    if (my_rank == 0) {
        std::string buffer(prefix);
        buffer.append("dets");
        size_t n_dets = read_dets(buffer, dets);
        
        buffer = prefix;
        buffer.append("vals");
        FILE *file_v = fopen(buffer.c_str(), "r");
        if (!file_v) {
            std::string msg("Could not find file:");
            msg.append(buffer);
            throw std::runtime_error(msg);
        }
        int num_read_v = 1;
        size_t n_vals = 0;
        
        int *val_arr = (int *) vals;
        while (num_read_v == 1) {
            num_read_v = fscanf(file_v, "%d\n", &val_arr[n_vals]);
            n_vals++;
        }
        n_vals--;
        if (n_vals > n_dets) {
            std::cerr << "Warning: fewer determinants than values read in\n";
            return n_dets;
        }
        else if (n_vals < n_dets) {
            std::cerr << "Warning: fewer values than determinants read in\n";
        }
        return n_vals;
    }
    else {
        return 0;
    }
}

size_t load_vec_txt(const std::string &prefix, Matrix<uint8_t> &dets, double *vals) {
    int my_rank = 0;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
#endif
    
    if (my_rank == 0) {
        std::string buffer(prefix);
        buffer.append("dets");
        size_t n_dets = read_dets(buffer, dets);
        
        buffer = prefix;
        buffer.append("vals");
        FILE *file_v = fopen(buffer.c_str(), "r");
        if (!file_v) {
            std::string msg("Could not find file:");
            msg.append(buffer);
            throw std::runtime_error(msg);
        }
        int num_read_v = 1;
        size_t n_vals = 0;
        
        double *val_arr = (double *)vals;
        while (num_read_v == 1) {
            num_read_v = fscanf(file_v, "%lf\n", &val_arr[n_vals]);
            n_vals++;
        }
        n_vals--;
        if (n_vals > n_dets) {
            std::cerr << "Warning: fewer determinants than values read in\n";
            return n_dets;
        }
        else if (n_vals < n_dets) {
            std::cerr << "Warning: fewer values than determinants read in\n";
        }
        return n_vals;
    }
    else {
        return 0;
    }
}


size_t read_dets(const std::string &path, Matrix<uint8_t> &dets) {
    int my_rank = 0;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
#endif
    
    if (my_rank == 0) {
        FILE *file_d = fopen(path.c_str(), "r");
        if (!file_d) {
            std::string msg("Could not find file: ");
            msg.append(path);
            throw std::runtime_error(msg);
        }
        
        int num_read_d = 1;
        size_t n_dets = 0;
        long long in_det;
        size_t max_size = dets.cols();
        
        while (num_read_d == 1) {
            num_read_d = fscanf(file_d, "%lld\n", &in_det);
            for (size_t byte_idx = 0; byte_idx < 8 && byte_idx < max_size; byte_idx++) {
                dets(n_dets, byte_idx) = in_det & 255;
                in_det >>= 8;
            }
            n_dets++;
        }
        return n_dets - 1;
    }
    else {
        return 0;
    }
}


void save_proc_hash(const std::string &path, unsigned int *proc_hash, size_t n_hash) {
    int my_rank = 0;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
#endif
    
    std::string buffer(path);
    buffer.append("hash.dat");
    if (my_rank == 0) {
        ofstream file_p(buffer, ios::binary);
        if (!file_p.is_open()) {
            std::string error("Could not save file at path ");
            error.append(buffer);
            throw std::runtime_error(error);
        }
        file_p.write((const char *)proc_hash, sizeof(unsigned int) * n_hash);
        file_p.close();
    }
}


void load_proc_hash(const std::string &path, unsigned int *proc_hash) {
    std::string buffer(path);
    buffer.append("hash.dat");
    ifstream file_p(buffer, ios::binary);
    if (!file_p.is_open()) {
        std::string error("Could not open saved hash scrambler at ");
        error.append(buffer);
        throw std::runtime_error(error);
    }
    file_p.read((char *)proc_hash, 1000);
    file_p.close();
}

void load_rdm(const std::string &path, double *vals) {
    FILE *file_p = fopen(path.c_str(), "r");
    if (!file_p) {
        fprintf(stderr, "Error: could not open RDM file.\n");
        return;
    }
    int num_read = 1;
    size_t n_vals = 0;
    while (num_read == 1) {
        num_read = fscanf(file_p, "%lf\n", &vals[n_vals]);
        n_vals++;
    }
    fclose(file_p);
}
