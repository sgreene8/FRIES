/*! \file
 *
 * \brief Utilities for reading/writing data from/to disk.
 */

#include "io_utils.hpp"
#include <FRIES/det_store.h>
#include <cmath>

size_t read_csv_line(std::ifstream &file, double *data) {
    size_t n_read = 0;
    std::string line;
    if (std::getline(file, line)) {
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

size_t read_csv_line(std::ifstream &file, uint8_t *data) {
    size_t n_read = 0;
    std::string line;
    if (std::getline(file, line)) {
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

size_t read_csv_line(std::ifstream &file, int *data) {
    size_t n_read = 0;
    std::string line;
    if (std::getline(file, line)) {
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

size_t read_csv(double *data, const std::string &fname) {
    std::ifstream in_f(fname);
    std::string line;
    size_t n_read = 0;
    size_t line_n_read = 1;
    while (line_n_read) {
        line_n_read = read_csv_line(in_f, &data[n_read]);
        n_read += line_n_read;
    }
    return n_read;
}

size_t read_csv(uint8_t *data, const std::string &fname) {
    std::ifstream in_f(fname);
    std::string line;
    size_t n_read = 0;
    size_t line_n_read = 1;
    while (line_n_read) {
        line_n_read = read_csv_line(in_f, &data[n_read]);
        n_read += line_n_read;
    }
    return n_read;
}

size_t read_csv(int *data, const std::string &fname) {
    std::ifstream in_f(fname);
    std::string line;
    size_t n_read = 0;
    size_t line_n_read = 1;
    while (line_n_read) {
        line_n_read = read_csv_line(in_f, &data[n_read]);
        n_read += line_n_read;
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
    read_csv(in_struct->symm, path);
    in_struct->symm = &(in_struct->symm[n_frz / 2]);
    
    path = hf_dir;
    path.append("hcore.txt");
    in_struct->hcore = new Matrix<double>(tot_orb, tot_orb);
    size_t n_read = read_csv((*(in_struct->hcore)).data(), path);
    if (n_read < tot_orb * tot_orb) {
        std::stringstream msg;
        msg << "Could not read " << tot_orb * tot_orb << " elements from " << path;
        throw std::runtime_error(msg.str());
    }
    
    path = hf_dir;
    path.append("eris.txt");
    in_struct->eris = new FourDArr(tot_orb, tot_orb, tot_orb, tot_orb);
    n_read = read_csv(in_struct->eris->data(), path);
    if (n_read < tot_orb * tot_orb * tot_orb * tot_orb) {
        std::stringstream msg;
        msg << "Could not read " << tot_orb * tot_orb * tot_orb * tot_orb << " elements from " << path;
        throw std::runtime_error(msg.str());
    }
}

fcidump_input *parse_fcidump(const std::string &hf_dir) {
    std::string path(hf_dir);
    path.append("sys_params.txt");
    std::ifstream in_file(path);
    if (!in_file.is_open()) {
        throw std::runtime_error("Could not open file sys_params.txt");
    }
    
    std::string keyword;
    std::getline(in_file, keyword); // skipping n_elec
    
    unsigned int n_frz;
    std::getline(in_file, keyword);
    std::getline(in_file, keyword);
    bool success = keyword == "n_frozen";
    if (success) {
        in_file >> n_frz;
        if (n_frz != 0) {
            throw std::runtime_error("n_frozen parameter is nonzero. Please set n_frozen to 0 and modify FCIDUMP file accordingly.");
        }
    }
    
    std::getline(in_file, keyword);
    std::getline(in_file, keyword); // skipping n_orb
    
    std::getline(in_file, keyword);
    std::getline(in_file, keyword);
    success = keyword == "eps";
    double eps;
    if (success) {
        in_file >> eps;
    }
    else {
        throw std::runtime_error("Fould not find eps parameter in sys_params.txt");
    }
    
    std::getline(in_file, keyword);
    std::getline(in_file, keyword);
    success = keyword == "hf_energy";
    double hf_en;
    if (success) {
        in_file >> hf_en;
    }
    else {
        throw std::runtime_error("Fould not find hf_energy parameter in sys_params.txt");
    }
    
    in_file.close();
    
    std::string file_line;
    
    path = hf_dir;
    path.append("FCIDUMP");
    in_file.open(path);
    std::getline(in_file, file_line);
    size_t n_orb_pos = file_line.find("NORB=");
    size_t n_elec_pos = file_line.find(",NELEC=");
    std::string substr = file_line.substr(n_orb_pos + 5, n_elec_pos - (n_orb_pos + 5));
    uint32_t n_orb;
    n_orb = std::stoi(substr);
    
    size_t ms_pos = file_line.find(",MS2=");
    substr = file_line.substr(n_elec_pos + 7, ms_pos - (n_elec_pos + 7));
    uint32_t n_elec = std::stoi(substr);
    
    substr = file_line.substr(ms_pos + 5, 1);
    if (std::stoi(substr) != 0) {
        throw std::runtime_error("MS2 is not zero in FCIDUMP file.");
    }
    
    fcidump_input *in_struct = new fcidump_input(n_orb);
    in_struct->eps = eps;
    in_struct->hf_en = hf_en;
    in_struct->n_elec = n_elec;
    in_struct->hcore = new Matrix<double>(n_orb, n_orb);
    
    std::getline(in_file, file_line);
    size_t orbsym_pos = file_line.find("ORBSYM=");
    substr = file_line.substr(orbsym_pos + 7, keyword.length() - 1 - orbsym_pos - 7);
    std::stringstream ss_line(substr);
//    size_t n_read = 0;
//    while (ss_line.good()) {
//        std::getline(ss_line, substr, ',');
//        if (substr.length() > 0) {
//            in_struct->symm[n_read] = atoi(substr.c_str());
//            n_read++;
//        }
//    }
//    if (n_read != in_struct->n_orb_) {
//        throw std::runtime_error("Number of irrep labels read in after ORBSYM in FCIDUMP file does not equal number of orbitals");
//    }
    path = hf_dir;
    path.append("symm.txt");
    read_csv(in_struct->symm, path);
    
    std::getline(in_file, file_line); // ISYM
    std::getline(in_file, file_line); // &END
    
    while (in_file.good()) {
        double integral;
        in_file >> integral;
        if (!in_file.good()) {
            break;
        }
        uint16_t orbs[4];
        in_file >> orbs[0];
        in_file >> orbs[1];
        in_file >> orbs[2];
        in_file >> orbs[3];
        
        if (orbs[0] == 0 && orbs[1] == 0 && orbs[2] == 0 && orbs[3] == 0) { // ecore
            in_struct->hf_en -= integral;
//            break;
        }
        else if (orbs[1] == 0 && orbs[2] == 0 && orbs[3] == 0) { // orbital energy, ignoring for now
            continue;
        }
        else if (orbs[2] == 0 && orbs[3] == 0) { // one-electron integral
            (*in_struct->hcore)(orbs[1] - 1, orbs[0] - 1) = (*in_struct->hcore)(orbs[0] - 1, orbs[1] - 1) = integral;
        }
        else {
            in_struct->eris.chemist_ordered(orbs[3] - 1, orbs[2] - 1, orbs[1] - 1, orbs[0] - 1) = integral;
        }
    }
    return in_struct;
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
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
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
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
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
            std::cerr << "Warning: fewer determinants (" << n_dets << ") than values (" << n_vals << ") read in\n";
            return n_dets;
        }
        else if (n_vals < n_dets) {
            std::cerr << "Warning: fewer values (" << n_vals << ") than determinants (" << n_dets << ") read in\n";
        }
        file_v.close();
        return n_vals;
    }
    else {
        return 0;
    }
}


size_t load_vec_dice(const std::string &path, Matrix<uint8_t> &dets, double *vals,
                     uint8_t state, uint32_t n_orb) {
    std::ifstream dice_file(path);
    if (!dice_file.is_open()) {
        std::string msg("Could not open file: ");
        msg.append(path);
        throw std::runtime_error(msg);
    }
    std::string file_line;
    std::string state_str("State :");
    uint8_t states_found = 0;
    while (!dice_file.eof() && states_found < state + 1) {
        std::getline(dice_file, file_line);
        if (std::equal(state_str.begin(), state_str.end(), file_line.begin())) {
            states_found++;
        }
    }
    if (states_found != (state + 1)) {
        return 0;
    }
    size_t idx = 0;
    while (!dice_file.eof()) {
        std::getline(dice_file, file_line);
        if (std::equal(state_str.begin(), state_str.end(), file_line.begin())) {
            break;
        }
        else {
            std::stringstream ss_line(file_line);
            size_t line_idx;
            ss_line >> line_idx;
            if (line_idx != idx) {
                std::stringstream msg("Misaligned indices when reading Dice file: ");
                msg << "read " << line_idx << ", expected " << idx << '\n';
                throw std::runtime_error(msg.str());
            }
            uint8_t *curr_det = dets[idx];
            std::fill(curr_det, curr_det + dets.cols(), 0);
            ss_line >> vals[idx];
            size_t curr_pos = ss_line.tellg();
            const char *c_line = ss_line.str().c_str();
            uint8_t orb_idx = 0;
            curr_pos++;
            while (c_line[curr_pos] != '\0') {
                switch (c_line[curr_pos]) {
                    case '2':
                        set_bit(curr_det, orb_idx + n_orb);
                    case 'a':
                        set_bit(curr_det, orb_idx);
                    case '0':
                        orb_idx++;
                        break;
                    case 'b':
                        set_bit(curr_det, orb_idx + n_orb);
                        orb_idx++;
                        break;
                }
                curr_pos++;
            }
            idx++;
        }
    }
    return idx;
}


size_t read_dets(const std::string &path, Matrix<uint8_t> &dets) {
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


void save_proc_hash(const std::string &path, unsigned int *proc_hash, size_t n_hash) {
    int my_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
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


size_t load_last_line(const std::string &path, double *vals) {
    // (see https://stackoverflow.com/questions/11876290/c-fastest-way-to-read-only-last-line-of-text-file)
    std::ifstream in_f(path);
    size_t n_read = 0;
    if (in_f.is_open()) {
        in_f.seekg(-1, std::ios_base::end);

        bool keepLooping = true;
        while(keepLooping) {
            char ch;
            in_f.get(ch);

            if((int)in_f.tellg() <= 1) {
                in_f.seekg(0);
                keepLooping = false;
            }
            else if(ch == '\n') {
                keepLooping = false;
            }
            else {
                in_f.seekg(-2, std::ios_base::cur);
            }
        }
        n_read = read_csv_line(in_f, vals);
    }
    return n_read;
}
