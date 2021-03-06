/*! \file
 *
 * \brief Utilities for Near-uniform compression of Hamiltonian
 *
 * The elements of the intermediate matrices in the Near-Uniform scheme are
 * defined based on symmetry relationships among the orbitals in the HF basis.
 * This file therefore also contains functions for creating lists of
 * symmetry-allowed orbitals
 */

#include "near_uniform.hpp"
#include <FRIES/Hamiltonians/molecule.hpp>

void count_symm_virt(unsigned int counts[][2], uint8_t *occ_orbs,
                     unsigned int n_elec, SymmInfo *symm) {
    unsigned int i;
    size_t n_orb = symm->symm_vec.size();
    for (i = 0; i < n_irreps; i++) {
        counts[i][0] = symm->symm_lookup(i, 0);
        counts[i][1] = symm->symm_lookup(i, 0);
    }
    for (i = 0; i < n_elec / 2; i++) {
        counts[symm->symm_vec[occ_orbs[i]]][0] -= 1;
    }
    for (; i < n_elec; i++) {
        counts[symm->symm_vec[occ_orbs[i] - n_orb]][1] -= 1;
    }
}


unsigned int bin_sample(unsigned int n, double p, std::mt19937 &mt_obj) {
    double rn;
    unsigned int success = 0;
    for (unsigned int i = 0; i < n; i++) {
        rn = mt_obj() / (1. + UINT32_MAX);
        success += rn < p;
    }
    return success;
}

unsigned int _choose_uint(std::mt19937 &mt_obj, unsigned int nmax) {
    // Choose an integer uniformly on the interval [0, nmax)
    return (mt_obj() / (1. + UINT32_MAX) * nmax);
}

orb_pair _tri_to_occ_pair(uint8_t *occ_orbs, unsigned int num_elec, unsigned int tri_idx) {
    // Use triangle inversion to convert an electron pair index into an orb_pair object
    orb_pair pair;
    unsigned int orb_idx1 = ((sqrt(tri_idx * 8. + 1) - 1) / 2);
    unsigned int orb_idx2 =  (tri_idx - orb_idx1 * (orb_idx1 + 1.) / 2);
    orb_idx1 += 1;
    pair.orb1 = occ_orbs[orb_idx1];
    pair.orb2 = occ_orbs[orb_idx2];
    pair.spin1 = orb_idx1 / (num_elec / 2);
    pair.spin2 = orb_idx2 / (num_elec / 2);
    return pair;
}

orb_pair _choose_occ_pair(uint8_t *occ_orbs, unsigned int num_elec, std::mt19937 &mt_obj) {
    // Randomly & uniformly choose a pair of occupied orbitals
    unsigned int rand_pair = _choose_uint(mt_obj, num_elec * (num_elec - 1) / 2);
    return _tri_to_occ_pair(occ_orbs, num_elec, rand_pair);
}

unsigned int _count_doub_virt(orb_pair occ, uint8_t *orb_irreps,
                              unsigned int n_orb, unsigned int num_elec,
                              unsigned int virt_counts[][2]) {
    // Count number of spin- and symmetry-allowed unoccupied orbitals
    // given a chosen pair of occupied orbitals
    unsigned int n_allow;
    uint8_t sym_prod = (orb_irreps[occ.orb1 % n_orb] ^
                              orb_irreps[occ.orb2 % n_orb]);
    int same_symm = sym_prod == 0 && occ.spin1 == occ.spin2;

    if (occ.spin1 == occ.spin2){
        n_allow = n_orb - num_elec / 2;
    }
    else {
        n_allow = 2 * n_orb - num_elec;
    }

    for (unsigned int i = 0; i < n_irreps; i++) {
        if (virt_counts[i ^ sym_prod][occ.spin2] == same_symm)
            n_allow -= virt_counts[i][occ.spin1];
        if (occ.spin1 != occ.spin2 && virt_counts[i ^ sym_prod][occ.spin1] == same_symm)
            n_allow -= virt_counts[i][occ.spin2];
    }
    return n_allow;
}

unsigned int _doub_choose_virt1(orb_pair occ, uint8_t *det,
                                uint8_t *irreps, unsigned int n_orb,
                                unsigned int virt_counts[][2],
                                unsigned int num_allowed,
                                std::mt19937 &mt_obj) {
    // Choose the first virtual orbital for a double excitation uniformly from
    // among the allowed orbitals
    int virt_choice = 0;
    unsigned int orbital;
    unsigned int a_spin, b_spin;
    uint8_t sym_prod = (irreps[occ.orb1 % n_orb] ^ irreps[occ.orb2 % n_orb]);
    unsigned int n_virt2, a_symm;

    if (num_allowed <= 3) { // choose the orbital index and then search for it
        virt_choice = _choose_uint(mt_obj, num_allowed);

        if (occ.spin1 == occ.spin2) {
            a_spin = occ.spin1;
            b_spin = a_spin;
        }
        else {
            a_spin = 0;
            b_spin = 1;
        }
        // begin search for virtual orbital
        orbital = 0;
        while (virt_choice >= 0 && orbital < n_orb) {
            // check that this orbital is unoccupied
            if (!read_bit(det, orbital + a_spin * n_orb)) {
                a_symm = irreps[orbital];
                n_virt2 = (virt_counts[sym_prod ^ a_symm][b_spin] -
                           (sym_prod == 0 && a_spin == b_spin));
                if (n_virt2 != 0) {
                    virt_choice -= 1;
                }
            }
            orbital += 1;
        }
        if (virt_choice >= 0) {
            // Different spins and orbital not found; keep searching
            a_spin = 1;
            b_spin = 0;
            while (virt_choice >= 0 && orbital < 2 * n_orb) {
                // check that this orbital is unoccupied
                if (!read_bit(det, orbital)) {
                    a_symm = irreps[orbital - n_orb];
                    n_virt2 = (virt_counts[sym_prod ^ a_symm][b_spin] -
                               (sym_prod == 0 && a_spin == b_spin));
                    if (n_virt2 != 0) {
                        virt_choice -= 1;
                    }
                }
                orbital += 1;
            }
            orbital -= n_orb;
        }
        virt_choice = orbital - 1 + a_spin * n_orb;
    }
    else {  // choose the orbital by rejection
        n_virt2 = 0;
        while (n_virt2 == 0) {
            if (occ.spin1 == occ.spin2) {
                a_spin = occ.spin1;
                b_spin = a_spin;
                virt_choice = _choose_uint(mt_obj, n_orb) + a_spin * n_orb;
            }
            else {
                virt_choice = _choose_uint(mt_obj, 2 * n_orb);
                a_spin = virt_choice / n_orb;
                b_spin = 1 - a_spin;
            }
            if (!read_bit(det, virt_choice)) { // check if unoccupied
                a_symm = irreps[virt_choice % n_orb];
                n_virt2 = (virt_counts[sym_prod ^ a_symm][b_spin] -
                           (sym_prod == 0 && a_spin == b_spin));
            }
        }
    }
    return virt_choice;
}


unsigned int _doub_choose_virt2(unsigned int spin_shift, uint8_t *det,
                                const uint8_t *symm_row,
                                unsigned int virt1, unsigned int n_allow,
                                std::mt19937 &mt_obj) {
    // Choose the second virtual orbial uniformly
    int orb_idx = _choose_uint(mt_obj, n_allow);
    unsigned int orbital = 0;
    unsigned int symm_idx = 1;
    // Search for chosen orbital
    while (orb_idx >= 0) {
        orbital = symm_row[symm_idx] + spin_shift;
        if (!read_bit(det, orbital) && orbital != virt1) {
            orb_idx -= 1;
        }
        symm_idx += 1;
    }
    return orbital;
}


unsigned int doub_multin(uint8_t *det, uint8_t *occ_orbs, unsigned int num_elec,
                         SymmInfo *symm,
                         unsigned int (* unocc_sym_counts)[2], unsigned int num_sampl,
                         std::mt19937 &mt_obj, uint8_t (* chosen_orbs)[4], double *prob_vec) {
    unsigned int i, a_symm, b_symm, a_spin, b_spin, sym_prod;
    unsigned int unocc1, unocc2, m_a_allow, m_a_b_allow, m_b_a_allow;
    orb_pair occ;
    unsigned int num_nonnull = 0;
    std::vector<uint8_t> &orb_symm = symm->symm_vec;
    uint32_t num_orb = (uint32_t) orb_symm.size();
    
    for (i = 0; i < num_sampl; i++) {
        occ = _choose_occ_pair(occ_orbs, num_elec, mt_obj);
        
        sym_prod = orb_symm[occ.orb1 % num_orb] ^ orb_symm[occ.orb2 % num_orb];
        
        m_a_allow = _count_doub_virt(occ, symm->symm_vec.data(), num_orb, num_elec, unocc_sym_counts);
        
        if (m_a_allow == 0)
            continue;
        
        unocc1 = _doub_choose_virt1(occ, det, symm->symm_vec.data(), num_orb, unocc_sym_counts, m_a_allow, mt_obj);
        a_spin = unocc1 / num_orb; // spin of 1st virtual orbital chosen
        b_spin = occ.spin1 ^ occ.spin2 ^ a_spin; // 2nd virtual spin
        a_symm = orb_symm[unocc1 % num_orb];
        b_symm = sym_prod ^ a_symm;
        
        m_a_b_allow = (unocc_sym_counts[b_symm][b_spin] - (sym_prod == 0 && a_spin == b_spin));
        
        // Choose second unoccupied orbital
        unocc2 = _doub_choose_virt2(b_spin * num_orb, det, symm->symm_lookup[b_symm],
                                    unocc1, m_a_b_allow, mt_obj);
        
        // Calculate probability of choosing this excitation
        m_b_a_allow = (unocc_sym_counts[a_symm][a_spin] - (sym_prod == 0 && a_spin == b_spin));
        
        prob_vec[num_nonnull] = 2. / num_elec / (num_elec - 1) / m_a_allow * (1. / m_a_b_allow
                                                                              + 1. / m_b_a_allow);
        
        chosen_orbs[num_nonnull][0] = occ.orb2;
        chosen_orbs[num_nonnull][1] = occ.orb1;
        if (unocc1 < unocc2) {
            chosen_orbs[num_nonnull][2] = unocc1;
            chosen_orbs[num_nonnull][3] = unocc2;
        }
        else {
            chosen_orbs[num_nonnull][2] = unocc2;
            chosen_orbs[num_nonnull][3] = unocc1;
        }
        num_nonnull++;
    }
    return num_nonnull;
}


unsigned int _sing_choose_occ(unsigned int *counts, unsigned int n_elec, std::mt19937 &mt_obj) {
    // Choose an occupied orbital with a nonzero number of symmetry-allowed excitations
    unsigned int elec_idx = 0, n_allow = 0;
    // Rejection sampling
    while (n_allow == 0){
        elec_idx = _choose_uint(mt_obj, n_elec);
        n_allow = counts[elec_idx];
    }
    return elec_idx;
}


unsigned int _sing_choose_virt(uint8_t *det, const uint8_t *symm_row,
                               unsigned int spin_shift, std::mt19937 &mt_obj
                               ) {
// Uniformly choose a virtual orbital with the specified symmetry
    int symm_idx = -1;
    unsigned int orbital = 0;
    // Rejection sampling
    while (symm_idx == -1) {
        symm_idx = _choose_uint(mt_obj, symm_row[0]);
        orbital = spin_shift + symm_row[symm_idx + 1];
        if (read_bit(det, orbital))
            symm_idx = -1; // orbital is occupied, choose again
    }
    return orbital;
}


unsigned int sing_multin(uint8_t *det, uint8_t *occ_orbs, unsigned int num_elec,
                         SymmInfo *symm,
                         unsigned int (* unocc_sym_counts)[2], unsigned int num_sampl,
                         std::mt19937 &mt_obj, uint8_t (* chosen_orbs)[2], double *prob_vec) {
    unsigned int delta_s, num_allowed, occ_orb, occ_symm, occ_spin;
    unsigned int m_allow[num_elec];
    std::vector<uint8_t> &orb_symm = symm->symm_vec;
    uint32_t num_orb = (uint32_t) orb_symm.size();
    
    delta_s = 0; // number of electrons with no symmetry-allowed excitations
    
    for (unsigned int elec_idx = 0; elec_idx < num_elec; elec_idx++) {
        occ_symm = orb_symm[occ_orbs[elec_idx] % num_orb];
        num_allowed = unocc_sym_counts[occ_symm][elec_idx / (num_elec / 2)];
        m_allow[elec_idx] = num_allowed;
        if (num_allowed == 0)
            delta_s++;
    }
    
    if (delta_s == num_elec) {
        return 0;
    }
    
    for (unsigned int j = 0; j < num_sampl; j++) {
        unsigned int elec_idx = _sing_choose_occ(m_allow, num_elec, mt_obj);
        occ_orb = occ_orbs[elec_idx];
        occ_symm = orb_symm[occ_orb % num_orb];
        occ_spin = occ_orb / num_orb;
        
        unsigned int virt_orb = _sing_choose_virt(det, symm->symm_lookup[occ_symm], occ_spin * num_orb, mt_obj);
        
        prob_vec[j] = (1. / m_allow[elec_idx] / (num_elec - delta_s));
        chosen_orbs[j][0] = occ_orb;
        chosen_orbs[j][1] = virt_orb;
    }
    return num_sampl;
}


unsigned int count_sing_allowed(uint8_t *occ_orbs, unsigned int num_elec,
                                uint8_t *orb_symm, unsigned int num_orb,
                                unsigned int (* unocc_sym_counts)[2]) {
    unsigned int elec_idx, num_allowed = 0;
    uint8_t occ_symm;
    for (elec_idx = 0; elec_idx < num_elec; elec_idx++) {
        occ_symm = orb_symm[occ_orbs[elec_idx] % num_orb];
        if (unocc_sym_counts[occ_symm][elec_idx / (num_elec / 2)] != 0)
            num_allowed++;
    }
    return num_allowed;
}


unsigned int count_sing_virt(uint8_t *occ_orbs, uint8_t num_elec,
                             uint8_t *orb_symm, uint8_t num_orb,
                             unsigned int (* unocc_sym_counts)[2],
                             uint8_t *occ_choice) {
    unsigned int num_allowed = 0;
    for (unsigned int elec_idx = 0; elec_idx < num_elec; elec_idx++) {
        uint8_t occ_symm = orb_symm[occ_orbs[elec_idx] % num_orb];
        unsigned int virt_allowed = unocc_sym_counts[occ_symm][elec_idx / (num_elec / 2)];
        if (virt_allowed != 0) {
            if (num_allowed == *occ_choice) {
                *occ_choice = elec_idx;
                return virt_allowed;
            }
            num_allowed++;
        }
    }
    return 0;
}


void symm_pair_wt(uint8_t *occ_orbs, unsigned int num_elec,
                  uint8_t *orb_symm, unsigned int num_orb,
                  unsigned int (* unocc_sym_counts)[2], uint8_t *occ_choice,
                  double *virt_weights, uint8_t *virt_counts) {
    orb_pair occ = _tri_to_occ_pair(occ_orbs, num_elec, *occ_choice);
    uint8_t sym_prod = orb_symm[occ.orb1 % num_orb] ^ orb_symm[occ.orb2 % num_orb];
    unsigned int m_a_allow = _count_doub_virt(occ, orb_symm, num_orb, num_elec, unocc_sym_counts);
    size_t symm_idx;
    if (m_a_allow == 0) {
        occ_choice[0] = 0;
        occ_choice[1] = 0;
        for (symm_idx = 0; symm_idx < n_irreps; symm_idx++) {
            virt_weights[symm_idx] = 0;
        }
        return;
    }
    occ_choice[0] = occ.orb2;
    occ_choice[1] = occ.orb1;
    unsigned int num_symm_pair, xor_row_idx, n_symm1, n_symm2;
    uint8_t xor_idx[4][8] = {
        {0, 1, 2, 3, 4, 5, 6, 7},
        {1, 3, 5, 7, 0, 0, 0, 0},
        {2, 3, 6, 7, 0, 0, 0, 0},
        {4, 5, 6, 7, 0, 0, 0, 0}
    };
    
    // Get pointer to list to use for enumerating symmetry products
    if ((occ.spin1 != occ.spin2) || sym_prod == 0) {
        num_symm_pair = n_irreps;
        xor_row_idx = 0;
    }
    else {
        num_symm_pair = n_irreps / 2;
        if (sym_prod == 1)
            xor_row_idx = 1;
        else if (sym_prod == 2 || sym_prod == 3)
            xor_row_idx = 2;
        else
            xor_row_idx = 3;
    }
    if (sym_prod == 0 && (occ.spin1 == occ.spin2)) {
        for (symm_idx = 0; symm_idx < num_symm_pair; symm_idx++) {
            n_symm1 = unocc_sym_counts[xor_idx[xor_row_idx][symm_idx]][occ.spin1];
            if (n_symm1 > 1) {
                virt_weights[symm_idx] = n_symm1 * 1. / m_a_allow;
                virt_counts[symm_idx] = n_symm1 * (n_symm1 - 1) / 2;
            }
            else
                virt_weights[symm_idx] = 0;
        }
    }
    else {
        for (symm_idx = 0; symm_idx < num_symm_pair; symm_idx++) {
            n_symm1 = unocc_sym_counts[xor_idx[xor_row_idx][symm_idx]][occ.spin1];
            n_symm2 = unocc_sym_counts[sym_prod ^ xor_idx[xor_row_idx][symm_idx]][occ.spin2];
            if (n_symm1 != 0 && n_symm2 != 0) {
                virt_weights[symm_idx] = 1. * (n_symm1 + n_symm2) / m_a_allow;
                virt_counts[symm_idx] = n_symm2 * n_symm1;
            }
            else {
                virt_weights[symm_idx] = 0;
            }
        }
    }
    for (symm_idx = num_symm_pair; symm_idx < n_irreps; symm_idx++) {
        virt_weights[symm_idx] = 0;
    }
}

uint8_t virt_from_idx(uint8_t *det, uint8_t *lookup_row,
                            uint8_t spin_shift, unsigned int index) {
    unsigned int symm_index;
    uint8_t orbital;
    for (symm_index = 0; symm_index < lookup_row[0]; symm_index++) {
        orbital = spin_shift + lookup_row[1 + symm_index];
        if (!read_bit(det, orbital)) {
            if (index == 0) {
                return orbital;
            }
            index--;
        }
    }
    return 255;
}
