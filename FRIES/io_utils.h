/*! \file
 *
 * \brief Utilities for reading/writing data from/to disk.
 */

#ifndef io_utils_h
#define io_utils_h

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <FRIES/Ext_Libs/csvparser.h>
#include <FRIES/mpi_switch.h>
#include <FRIES/vec_utils.h>


#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Read an array of floating-point numbers from a .csv file
 *
 * \param [out] buf     Array in which read-in numbers are stored
 * \param [in] fname    Path of file
 * \returns          Total number of values read from the file
 */
size_t read_doub_csv(double *buf, char *fname);


/*! \brief Read an array of unsigned bytes from a .csv file
 *
 * \param [out] buf     Array in which read-in numbers are stored
 * \param [in] fname    Path of file
 * \returns          Total number of values read from the file 
 */
size_t read_uchar_csv(unsigned char *buf, char *fname);


/*! \brief Data structure containing the output of a Hartree-Fock calculation */
typedef struct {
    unsigned int n_elec; ///< Total number of electrons (including frozen) in the system
    unsigned int n_frz; ///< Suggested number of core electrons to freeze
    unsigned int n_orb; ///< Number of spatial orbitals in the HF basis
    double *eris; ///< Pointer to array of 2-electron integrals
    double *hcore; ///< Pointer to array of 1-electron integrals
    double hf_en; ///< HF electronic energy
    double eps; ///< Suggested imaginary time step to use in DMC calculations
    unsigned char *symm; ///< Irreps of orbitals in the HF basis
} hf_input;


/*! \brief Data structure describing the parameters of a Hubbard-Holstein
 * calculation
 */
typedef struct {
    unsigned int n_elec; ///< Total number of electrons in the system
    unsigned int lat_len; ///< Number of sites along one dimension of the lattice
    unsigned int n_dim; ///< Dimensionality of the lattice
    double elec_int; ///< On-site repulsion term
    double eps; ///< Suggested imaginary time step to use in DMC calculations
    double hf_en; ///< HF electronic energy
} hh_input;


/*! \brief Read in parameters from a Hartree-Fock calculation:
 * total number of electrons
 * number of (unfrozen) orbitals
 * number of frozen (core) electrons
 * imaginary time step (eps)
 * HF electronic energy
 * matrix of one-electron integrals
 * tensor of two-electron integrals
 *
 * \param [in] hf_dir       Path to directory where the files sys_params.txt,
 *                          eris.txt, hcore.txt, and symm.txt are stored
 * \param [in] in_struct    Structure where the data will be stored
 * \return 0 if successful, -1 if unsuccessful
 */
int parse_hf_input(const char *hf_dir, hf_input *in_struct);


/*! \brief Read in the following parameters for a Hubbard-Holstein calculation
 * total number of electrons
 * number of sites along one dimension of the lattice
 * dimensionality of the lattice
 * electron-electron interaction term, U
 * imaginary time step (eps)
 * HF electronic energy
 *
 * \param [in] hh_path      Path to the file containing the Hubbard-Holstein
 *                          parameters
 * \param [in] in_struct    Structure where the data will be stored
 * \return 0 if successful, -1 if unsuccessful
 */
int parse_hh_input(const char *hh_path, hh_input *in_struct);


/*! \brief Load a sparse vector in .txt format from disk
 *
 * This function can read from a number of files that is <= the number of MPI
 * processes
 *
 * \param [in] prefix       prefix of files containing the vector, including the
 *                          directory. File names should be in the format
 *                          [prefix]dets[i].txt and [prefix]vals[i].txt, where i
 *                          is an index starting at 00
 * \param [out] dets        Array of element indices read in
 * \param [out] vals        Array of element values read in
 * \param [in] type         Data type of the vector
 * \return total number of elements in the vector
 */
size_t load_vec_txt(const char *prefix, long long *dets, void *vals, dtype type);


/*! \brief Save to disk the array of random numbers used to assign Slater
 * determinants to MPI processes
 *
 * This function only does something on the 0th MPI process. The numbers are
 * saved in binary format
 *
 * \param [in] path         File will be saved at the path [path]hash.dat
 * \param [in] proc_hash    Array of random numbers to save (length \p n_hash)
 */
void save_proc_hash(const char *path, unsigned int *proc_hash, size_t n_hash);


/*! \brief Load from disk the array of random numbers used to assign Slater
 * determinants to MPI processes
 *
 * The random numbers are loaded onto all MPI processes
 *
 * \param [in] path         File will be loaded from the path [path]hash.dat
 * \param [out] proc_hash   Numbers loaded from disk
 */
void load_proc_hash(const char *path, unsigned int *proc_hash);

    
#ifdef __cplusplus
}
#endif

#endif /* io_utils_h */
