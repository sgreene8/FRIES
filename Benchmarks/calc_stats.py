#!/usr/bin/env python2

import numpy
import emcee

# Exact energies
#ecorr =-128.709476 + 128.49634973054 # Ne
#ecorr = -109.276527 + 108.95454605585 # N2
#ecorr = -108.96695 + 108.222900457162 # N2 stretched
#ecorr = -0.2178339712 # H2O

burn_in = 40000

root_dir = "./" # path to the output files from the calculation
proj_num = numpy.genfromtxt(root_dir + 'projnum.txt')
proj_den = numpy.genfromtxt(root_dir + 'projden.txt')
traj_len = min(proj_den.shape[0], proj_num.shape[0])

proj_num = proj_num[burn_in:traj_len]
proj_den = proj_den[burn_in:traj_len]
num_mean = numpy.mean(proj_num)
den_mean = numpy.mean(proj_den)

corr_traj = proj_num / den_mean - num_mean * proj_den / den_mean**2

iat = emcee.autocorr.integrated_time(corr_traj, c=2)
print('iat: ' + str(iat))

var = numpy.var(corr_traj)
ediff = (num_mean / den_mean - ecorr) * 1e3
se = numpy.sqrt(var * iat / proj_den.shape[0])
std_err = se * 1e3
effs = 1. / var / iat
print("Mean error ± 2 sigma (millihartrees) = {0:.2f} ± {1:.2f}".format(ediff, 2 * std_err))
print("Efficiency: " + str(effs))

