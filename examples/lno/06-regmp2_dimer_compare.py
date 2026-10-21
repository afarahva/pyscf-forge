#!/usr/bin/env python
r'''
kappa-regularized MP2 (kappa-MP2) in LNO-CCSD: water dimer comparison.

kappa-MP2 (arXiv:2508.15744) damps each MP2 amplitude by a single factor
    g_ijab = 1 - exp(-kappa * (e_a + e_b - e_i - e_j)),    kappa = 1.1 (default)
The LNO base class has two low-level amplitude settings (see LNO.__init__):
    lno_amp : amplitudes for the LNO density matrices (i.e. when choosing the
              local natural orbital active space); default 'mp2'
    pt2_amp : amplitudes for the fragment MP2 energy and the CCSD initial
              guess; default None (same as lno_amp)
lno_amp='kappa-mp2' alone uses kappa-MP2 everywhere.

This script runs LNO-CCSD on a water dimer at lno_thresh = [1e-5, 1e-6] with
PM-localized orbitals and compares three calculations:

  1. plain MP2 guess / plain LNO-DM   (baseline)
  2. kappa-MP2 CC guess (pt2_amp='kappa-mp2')    -- expected within ~1 mEh of
     baseline: the converged CCSD energy should barely depend on the starting
     amplitudes.
  3. kappa-MP2 LNO-DM  (lno_amp='kappa-mp2', pt2_amp='mp2') -- expected close,
     but possibly a little further from baseline, since the selected active
     space is not identical.
'''

import numpy as np
from pyscf import gto, scf, lno, lo

atom = '''
O   0.000000 0.000000  0.000000
H   0.758602 0.000000  0.504284
H   0.260455 0.000000 -0.872893
O   3.000000 0.500000  0.000000
H   3.758602 0.500000  0.504284
H   3.260455 0.500000 -0.872893
'''
mol = gto.M(atom=atom, basis='cc-pvdz', spin=0, verbose=3, max_memory=8000)
mf = scf.RHF(mol).density_fit().run()

# PM-localized occupied orbitals; each LO is its own fragment
orbocc = mf.mo_coeff[:, mf.mo_occ > 1e-6]
nocc = orbocc.shape[1]
lo_coeff = lo.PipekMezey(mol, orbocc).kernel()
frag_lolist = [[i] for i in range(nocc)]

lno_thresh = [1e-5, 1e-6]
kappa = 1.1

def run(**kw):
    mcc = lno.LNOCCSD(mf, lo_coeff, frag_lolist, lno_thresh=lno_thresh)
    mcc.kappa = kappa
    for k, v in kw.items():
        setattr(mcc, k, v)
    mcc.kernel()
    return mcc.e_corr_ccsd

e_base = run()
e_cc   = run(pt2_amp='kappa-mp2')                 # only the CC guess / MP2 energy
e_lno  = run(lno_amp='kappa-mp2', pt2_amp='mp2')  # only the LNO density matrices

print()
print('LNO-CCSD correlation energy (water dimer, lno_thresh=%s, kappa=%.2f)'
      % (lno_thresh, kappa))
print('  plain MP2 guess / plain LNO-DM : % .10f' % e_base)
print('  kappa-MP2 CC guess (pt2_amp)  : % .10f  (dE = % .3e Eh)'
      % (e_cc,  e_cc  - e_base))
print('  kappa-MP2 LNO-DM  (lno_amp)  : % .10f  (dE = % .3e Eh)'
      % (e_lno, e_lno - e_base))
print()
print('Expectation: |dE(pt2_amp)| < 1 mEh (CCSD is insensitive to its guess);')
print('             lno_amp is also close but may differ a bit more, since')
print('             the selected LNO active space is not exactly the same.')

assert abs(e_cc - e_base) < 1e-3, 'kappa-MP2 CC guess changed CCSD by > 1 mEh'
