#!/usr/bin/env python
r'''
kappa-regularized MP2 (kappa-MP2) in restricted LNO-CCSD: behavior of the
regmp2 switches.

See LNO.__init__ for the switches (regmp2_lno, regmp2_cc, regmp2) and
examples/lno/06-regmp2_dimer_compare.py for the physical motivation.

This script sanity-checks the implementation on a single water molecule:
  * with a huge kappa the damping factor g -> 1, so regmp2=True must reproduce
    the plain (flags-off) result exactly;
  * regmp2 (everything) equals switching on regmp2_lno and regmp2_cc together;
  * regmp2_cc changes the reported fragment MP2 energy (kappa-MP2 energy),
    while regmp2_lno leaves it essentially unchanged (only the LNO active-space
    selection is affected).
'''

import numpy as np
from pyscf import gto, scf, lno, lo

mol = gto.M(atom='''O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24''',
            basis='cc-pvdz', verbose=3)
mf = scf.RHF(mol).density_fit().run()
frozen = 1

orbocc = mf.mo_coeff[:, frozen:np.count_nonzero(mf.mo_occ)]
lo_coeff = lo.PipekMezey(mol, orbocc).kernel()
frag_lolist = [[i] for i in range(lo_coeff.shape[1])]

def run(**kw):
    mcc = lno.LNOCCSD(mf, lo_coeff, frag_lolist, frozen=frozen)
    mcc.lno_thresh = [1e-4, 1e-5]
    for k, v in kw.items():
        setattr(mcc, k, v)
    mcc.kernel()
    return mcc.e_corr_pt2, mcc.e_corr_ccsd

base = run()
big  = run(regmp2=True, kappa=1e6)
full = run(regmp2=True, kappa=1.1)
lno_only = run(regmp2_lno=True, kappa=1.1)
cc_only  = run(regmp2_cc=True,  kappa=1.1)
both     = run(regmp2_lno=True, regmp2_cc=True, kappa=1.1)

print()
print('                              mp2              ccsd')
print('baseline (flags off)     % .10f  % .10f' % base)
print('regmp2=True   kappa=1e6  % .10f  % .10f' % big)
print('regmp2=True   kappa=1.1  % .10f  % .10f' % full)
print('regmp2_lno    kappa=1.1  % .10f  % .10f' % lno_only)
print('regmp2_cc     kappa=1.1  % .10f  % .10f' % cc_only)
print('lno+cc flags  kappa=1.1  % .10f  % .10f' % both)

assert np.allclose(base, big,  atol=1e-8), 'large-kappa must match baseline'
assert np.allclose(full, both, atol=1e-9), 'regmp2 must equal (lno & cc)'
assert not np.isclose(cc_only[0], base[0], atol=1e-6), 'regmp2_cc must change MP2 energy'
print('\nall checks passed')
