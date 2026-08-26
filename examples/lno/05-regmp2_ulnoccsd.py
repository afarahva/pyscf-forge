#!/usr/bin/env python
r'''
kappa-regularized MP2 (kappa-MP2) in unrestricted LNO-CCSD.

Same regmp2 switches as the restricted case (see LNO.__init__ and
examples/lno/04-regmp2_lnoccsd.py). This script checks the UHF path on the
OH radical:
  * with a huge kappa (g -> 1) regmp2=True reproduces the flags-off result;
  * regmp2_cc changes the reported fragment MP2 energy (kappa-MP2 energy).
'''

import numpy as np
from pyscf import gto, scf, lno, lo

mol = gto.M(atom='''O 0 0 0; H 0 0 0.97''', basis='cc-pvdz', spin=1, verbose=3)
mf = scf.UHF(mol).density_fit().run()
frozen = 1

orbocc, lo_coeff = [], []
for s in range(2):
    oc = mf.mo_coeff[s][:, frozen:np.count_nonzero(mf.mo_occ[s])]
    orbocc.append(oc)
    lo_coeff.append(lo.PipekMezey(mol, oc).kernel())
oa = [[[i], []] for i in range(orbocc[0].shape[1])]
ob = [[[], [i]] for i in range(orbocc[1].shape[1])]
frag_lolist = oa + ob

def run(**kw):
    mcc = lno.ULNOCCSD(mf, lo_coeff, frag_lolist, frozen=frozen)
    mcc.lno_thresh = [1e-4, 1e-5]
    for k, v in kw.items():
        setattr(mcc, k, v)
    mcc.kernel()
    return mcc.e_corr_pt2, mcc.e_corr_ccsd

base = run()
big  = run(regmp2=True, kappa=1e6)
full = run(regmp2=True, kappa=1.1)
cc_only = run(regmp2_cc=True, kappa=1.1)

print()
print('                              mp2              ccsd')
print('UHF baseline             % .10f  % .10f' % base)
print('UHF regmp2   kappa=1e6   % .10f  % .10f' % big)
print('UHF regmp2   kappa=1.1   % .10f  % .10f' % full)
print('UHF regmp2_cc kappa=1.1  % .10f  % .10f' % cc_only)

assert np.allclose(base, big, atol=1e-8), 'large-kappa must match baseline'
assert not np.isclose(cc_only[0], base[0], atol=1e-6), 'regmp2_cc must change MP2 energy'
print('\nall checks passed')
