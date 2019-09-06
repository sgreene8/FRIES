Fast Randomized Iteration for Electronic Structure (FRIES)
==========================================================

## Summary

This project contains implementations of various methods within the Fast Randomized Iteration (FRI) framework for performing Full Configuration Interaction calculations on molecular systems and the Hubbard model. A detailed description of the FRI framework for quantum chemistry problems can be found in this publication: (https://doi.org/10.1021/acs.jctc.9b00422) Implementations of the related Full Configuration Interaction (FCIQMC) method (http://dx.doi.org/10.1063/1.3193710) are also included for comparison.

In their current form, these FRI methods evolve an arbitrary initial vector in Slater determinant space towards the ground-state eigenvector via a stochastic implementation of the power method. Efficiency is achieved by stochastically compressing the Hamiltonian matrix and solution vector at each iteration and exploiting sparse linear algebra techniques. Observables are calculated by averaging over many iterations.

## Acknowledgments

This work was supported by startup funding provided by the University of Chicago and the Flatiron Institute, a division of the Simons Foundation. Additional funding was provided by the Molecular Software Sciences Institute (MolSSI) through a software "seed" fellowship, the Advanced Scientific Computing Research program through award DE-SC0014205, NSF RTG awards 1547396 and 1646339, and a MacCracken Fellowship from New York University.
