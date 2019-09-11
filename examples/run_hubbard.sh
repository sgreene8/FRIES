#!/bin/sh

#  Example of performing calculations on the 1-D Hubbard model at half-filling
# with six sites

# FCIQMC
../build/FRIES_bin/fciqmc_hh --params_path hubbard_params.txt --target 100000 --max_dets 1000

# Systematic FCI-FRI
../build/FRIES_bin/frisys_hh --params_path hubbard_params.txt --target 1000 --max_dets 1000 --vec_nonz 200 --mat_nonz 500
