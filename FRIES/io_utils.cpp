/*! \file
 *
 * \brief Utilities for reading/writing data from/to disk.
 */

#include "io_utils.hpp"

size_t read_csv(double *data, const std::string &fname) {
    std::ifstream in_f(fname);
    std::string line;
    size_t n_read = 0;
    while (std::getline(in_f, line)) {
        std::stringstream ss_line(line);
        while (ss_line.good()) {
            std::string substr;
            std::getline(ss_line, substr, ',');
            std::stringstream number(substr);
            number << std::scientific;
            number >> data[n_read];
            n_read++;
        }
    }
    return n_read;
}

size_t read_csv(uint8_t *data, const std::string &fname) {
    std::ifstream in_f(fname);
    std::string line;
    size_t n_read = 0;
    while (std::getline(in_f, line)) {
        std::stringstream ss_line(line);
        while (ss_line.good()) {
            std::string substr;
            std::getline(ss_line, substr, ',');
            std::stringstream number(substr);
            uint16_t inp;
            number >> inp;
            data[n_read] = inp;
            n_read++;
        }
    }
    return n_read;
}

size_t read_csv(int *data, const std::string &fname) {
    std::ifstream in_f(fname);
    std::string line;
    size_t n_read = 0;
    while (std::getline(in_f, line)) {
        std::stringstream ss_line(line);
        while (ss_line.good()) {
            std::string substr;
            std::getline(ss_line, substr, ',');
            std::stringstream number(substr);
            number >> data[n_read];
            n_read++;
        }
    }
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
        std::ifstream file_v(buffer);
        if (!file_v.is_open()) {
            std::string msg("Could not find file: ");
            msg.append(buffer);
            throw std::runtime_error(msg);
        }
        size_t n_vals = 0;
        
        while (file_v >> vals[n_vals]) {
            n_vals++;
        }
        if (n_vals > n_dets) {
            std::cerr << "Warning: fewer determinants than values read in\n";
            return n_dets;
        }
        else if (n_vals < n_dets) {
            std::cerr << "Warning: fewer values than determinants read in\n";
        }
        file_v.close();
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
        std::ifstream file_v(buffer);
        if (!file_v.is_open()) {
            std::string msg("Could not open file: ");
            msg.append(buffer);
            throw std::runtime_error(msg);
        }
        size_t n_vals = 0;
        
        while (file_v >> vals[n_vals]) {
            n_vals++;
        }
        if (n_vals > n_dets) {
            std::cerr << "Warning: fewer determinants than values read in\n";
            return n_dets;
        }
        else if (n_vals < n_dets) {
            std::cerr << "Warning: fewer values than determinants read in\n";
        }
        file_v.close();
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
        std::ifstream file_d(path);
        if (!file_d.is_open()) {
            std::string msg("Could not open file: ");
            msg.append(path);
            throw std::runtime_error(msg);
        }
        
        size_t n_dets = 0;
        long long in_det;
        size_t max_size = dets.cols();
        
        while (file_d >> in_det) {
            for (size_t byte_idx = 0; byte_idx < 8 && byte_idx < max_size; byte_idx++) {
                dets(n_dets, byte_idx) = in_det & 255;
                in_det >>= 8;
            }
            n_dets++;
        }
        file_d.close();
        return n_dets;
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
        std::ofstream file_p(buffer, std::ios::binary);
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
    std::ifstream file_p(buffer, std::ios::binary);
    if (!file_p.is_open()) {
        std::string error("Could not open saved hash scrambler at ");
        error.append(buffer);
        throw std::runtime_error(error);
    }
    file_p.read((char *)proc_hash, 1000);
    file_p.close();
}

void load_rdm(const std::string &path, double *vals) {
    std::ifstream file_p(path);
    if (!file_p.is_open()) {
        std::string msg("Could not open file: ");
        msg.append(path);
        throw std::runtime_error(msg);
    }
    size_t n_vals = 0;
    while (file_p >> vals[n_vals]) {
        n_vals++;
    }
    file_p.close();
}
