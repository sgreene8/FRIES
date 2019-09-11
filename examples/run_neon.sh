#!/bin/sh

# Examples demonstrating how to perform calculations on the Neon atom in the
# aug-cc-pVDZ basis, using the files in the Neon_augccpdvz_HF directory.
# Assuming that the build folder is located at ../build and that executables
# were compiled without MPI support

# Perform i-FCIQMC calculation with ~250k walkers
../build/FRIES_bin/fciqmc_mol --hf_path Neon_augccpdvz_HF/ --target 250000 --distribution NU --max_dets 300000 --initiator 3

# Perform systematic FCI-FRI calculation, compressing Hamiltonian to 260k elements and solution vector to 242k elements
../build/FRIES_bin/frisys_mol --hf_path Neon_augccpdvz_HF/ --target 260000 --vec_nonz 242000 --mat_nonz 260000 --max_dets 500000 --ini_dir Neon_augccpdvz_HF/
