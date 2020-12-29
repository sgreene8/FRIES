"""
This is an example script to demonstrate how to use pyscf to generate the input files needed 
to run an FRI excited-state calculation.
"""

import numpy as np
from pyscf import gto, scf, fci, mcscf

mol = gto.M(atom='Ne 0. 0. 0.', basis='aug-cc-pvdz', symmetry='d2h')
nfrz = 2 # Number of core electrons to freeze
mf = scf.RHF(mol).run()
hf_elec_e = mf.energy_elec()[0]
irreps = scf.hf_symm.get_orbsym(mol, mf.mo_coeff)
norb = mf.mo_coeff.shape[1]

from pyscf import ao2mo
eris = ao2mo.full(mol.intor('int2e_sph', aosym='s4'),mf.mo_coeff, compact=False)
eris.shape = (norb, norb, norb, norb)
eris = eris.transpose(0, 2, 1, 3).copy()
eris.shape = (norb**2, norb**2)

# Save Hartree-Fock output to disk
np.savetxt('eris.txt', eris, delimiter=',')
h_core = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
np.savetxt('hcore.txt', h_core, delimiter=',')
irreps = scf.hf_symm.get_orbsym(mol, mf.mo_coeff)
np.savetxt('symm.txt', irreps, fmt='%d')

norb = 11 # Truncate the virtual space for a CASCI calculation
nelec = mol.nelectron

n_states = 10
half_elec = (nelec - nfrz) // 2
mycas = mcscf.CASCI(mf, norb - nfrz//2, (half_elec, half_elec))
mycas.fcisolver.nroots = n_states

e_fci, e_cas, c_fci, casci_coef, casci_1e = mycas.kernel()
for i, x in enumerate(c_fci):
    print('state %d, E = %.12f  2S+1 = %.7f symm = %s' %
          (i, e_fci[i] - hf_elec_e, fci.spin_op.spin_square0(x, norb - nfrz//2, (half_elec, half_elec))[1], 
           fci.addons.guess_wfnsym(x, norb - nfrz//2, (half_elec, half_elec), irreps[nfrz//2:norb])))

# Save trial vectors to disk
fci_strings = fci.cistring.make_strings(range(norb - nfrz//2), nelec//2 - nfrz//2)
for idx in range(n_states):
    nonz = np.nonzero(np.abs(c_fci[idx]))
    dets = fci_strings[nonz[0]] + (fci_strings[nonz[1]] << (mf.mo_coeff.shape[1] - nfrz // 2))
    np.savetxt('trial_vecs/casci0' + str(idx) + 'dets', dets, fmt='%d')
    np.savetxt('trial_vecs/casci0' + str(idx) + 'vals', c_fci[idx][nonz])

