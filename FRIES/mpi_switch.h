/*! \file
 * \brief If USE_MPI is defined here, executable should be compiled with MPI
 */

#ifndef mpi_switch_h
#define mpi_switch_h

/*! \brief Flag indicating whether executable should be compiled with MPI */
#ifdef USE_MPI
#include "mpi.h"
#endif

#endif /* mpi_switch_h */
